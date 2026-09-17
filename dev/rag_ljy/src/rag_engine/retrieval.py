"""BM25+dense retrieval and client-side reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .stores.base import RetrievalStore


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
) -> list[dict[str, Any]]:
    bm25_hits = store.bm25_search(query, size=retriever_top_k)
    dense_hits = store.dense_search(
        query_vector,
        size=retriever_top_k,
        num_candidates=num_candidates,
    )
    return reciprocal_rank_fusion(
        {"bm25": bm25_hits, "dense": dense_hits},
        rank_constant=rank_constant,
        limit=coarse_top_k,
    )
