"""BM25+dense retrieval and client-side reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any

from .timing import measure_stage

if TYPE_CHECKING:
    from .stores.base import RetrievalStore


EXECUTION_MODES = frozenset({"sequential", "concurrent"})


def validate_execution_mode(mode: str) -> None:
    if mode not in EXECUTION_MODES:
        raise ValueError(f"execution_mode must be sequential or concurrent; got {mode!r}")


def _concurrent_hits(
    store: RetrievalStore, query: str, query_vector: list[float],
    retriever_top_k: int, num_candidates: int, executor: ThreadPoolExecutor,
    timings: dict[str, float] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Worker-local dictionaries avoid concurrent writes to the caller's timings.
    sparse_timings = {} if timings is not None else None
    dense_timings = {} if timings is not None else None

    def sparse():
        with measure_stage(sparse_timings, "bm25_ms"):
            return store.bm25_search(query, size=retriever_top_k)

    def dense():
        with measure_stage(dense_timings, "dense_ms"):
            return store.dense_search(
                query_vector, size=retriever_top_k, num_candidates=num_candidates,
            )

    futures = []
    try:
        # Submit BOTH before waiting. Only database calls run on these threads.
        futures.append(executor.submit(sparse))
        futures.append(executor.submit(dense))
    finally:
        # Drain both even if one fails or submission of the second fails. Never
        # return partial results or leave a sibling search running into next query.
        try:
            wait(futures)
        finally:
            if timings is not None:
                timings.update(sparse_timings or {})
                timings.update(dense_timings or {})
    return futures[0].result(), futures[1].result()


def reciprocal_rank_fusion(
    result_sets: Mapping[str, Sequence[dict[str, Any]]],
    rank_constant: int = 60,
    limit: int = 30,
) -> list[dict[str, Any]]:
    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1")
    if limit < 1:
        raise ValueError("limit must be at least 1")

    fused: dict[str, dict[str, Any]] = {}
    for channel, hits in result_sets.items():
        for rank, hit in enumerate(hits, start=1):
            item = fused.setdefault(
                hit["_id"],
                {
                    "_id": hit["_id"],
                    "_source": hit["_source"],
                    "_rrf_score": 0.0,
                    "_retrieval_ranks": {},
                    "_retrieval_scores": {},
                },
            )
            item["_rrf_score"] += 1.0 / (rank_constant + rank)
            item["_retrieval_ranks"][channel] = rank
            item["_retrieval_scores"][channel] = hit["_score"]

    return sorted(
        fused.values(),
        key=lambda hit: (-hit["_rrf_score"], hit["_id"]),
    )[:limit]


def hybrid_search(
    store: RetrievalStore,
    query: str,
    query_vector: list[float],
    retriever_top_k: int = 50,
    coarse_top_k: int = 30,
    num_candidates: int = 100,
    rank_constant: int = 60,
    timings: dict[str, float] | None = None,
    execution_mode: str = "sequential",
    executor: ThreadPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    validate_execution_mode(execution_mode)
    # Measure the actual interval including submission/join overhead, not an
    # estimated sum/max. Production pipelines/benchmarks reuse their own pool.
    with measure_stage(timings, "search_ms"):
        if execution_mode == "concurrent":
            if executor is None:
                # Convenience for standalone callers; its pool overhead is timed.
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-search") as pool:
                    bm25_hits, dense_hits = _concurrent_hits(
                        store, query, query_vector, retriever_top_k,
                        num_candidates, pool, timings,
                    )
            else:
                bm25_hits, dense_hits = _concurrent_hits(
                    store, query, query_vector, retriever_top_k,
                    num_candidates, executor, timings,
                )
        else:
            with measure_stage(timings, "bm25_ms"):
                bm25_hits = store.bm25_search(query, size=retriever_top_k)
            with measure_stage(timings, "dense_ms"):
                dense_hits = store.dense_search(
                    query_vector,
                    size=retriever_top_k,
                    num_candidates=num_candidates,
                )
    with measure_stage(timings, "rrf_ms"):
        return reciprocal_rank_fusion(
            {"bm25": bm25_hits, "dense": dense_hits},
            rank_constant=rank_constant,
            limit=coarse_top_k,
        )
