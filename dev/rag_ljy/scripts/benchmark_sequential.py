#!/usr/bin/env python3
"""Measure sequential or concurrent BM25/dense latency, optionally including models."""

from __future__ import annotations

import argparse
import os
import platform
import uuid
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter_ns

from _bootstrap import load_project_env

load_project_env()

from rag_engine import Settings, create_retrieval_store  # noqa: E402
from rag_engine.benchmark import run_benchmark  # noqa: E402
from rag_engine.benchmark_data import BenchmarkQuery, load_queries  # noqa: E402
from rag_engine.embedding import Qwen3Embedder  # noqa: E402
from rag_engine.query_runtime import create_pipeline  # noqa: E402
from rag_engine.retrieval import hybrid_search  # noqa: E402
from rag_engine.timing import measure_stage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--queries", type=Path, help="query_id/query or BEIR _id/text JSONL")
    inputs.add_argument("--query", action="append", help="one-off smoke query; repeat the option for several")
    parser.add_argument("--backend", choices=["elasticsearch", "milvus"])
    parser.add_argument("--device", required=True, help="allocated device, e.g. npu:3")
    parser.add_argument("--scope", choices=["retrieval-only", "full-query"], default="retrieval-only")
    parser.add_argument("--execution-mode", choices=["sequential", "concurrent"], default="sequential", help="benchmark default preserves the sequential baseline")
    parser.add_argument("--es-index", default="nfcorpus-chunks-v1")
    parser.add_argument("--milvus-collection", default="nfcorpus_chunks_v1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, help="use the first N input queries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retriever-top-k", type=int, default=50)
    parser.add_argument("--coarse-top-k", type=int, choices=range(20, 51), default=30)
    parser.add_argument("--fine-top-k", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/rag-ljy-benchmarks"), help="parent for a new uniquely named run directory")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or (args.limit is not None and args.limit < 1):
        parser.error("warmup must be >=0; repeats and limit must be positive")
    if min(args.retriever_top_k, args.embedding_batch_size, args.reranker_batch_size) < 1:
        parser.error("Top-K and batch sizes must be positive")
    if args.query and any(not query.strip() for query in args.query):
        parser.error("queries cannot be empty")
    return args


def package_versions() -> dict[str, str | None]:
    result = {}
    for package in ("torch", "torch-npu", "transformers", "elasticsearch", "pymilvus"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _run(args: argparse.Namespace, lifecycle: ExitStack) -> None:
    if args.backend:
        os.environ["RETRIEVAL_BACKEND"] = args.backend
    settings = replace(
        Settings.from_env(), elasticsearch_index_name=args.es_index,
        milvus_collection_name=args.milvus_collection,
    )
    queries = load_queries(args.queries) if args.queries else [
        BenchmarkQuery(f"manual-{index}", query) for index, query in enumerate(args.query, 1)
    ]
    if args.limit is not None:
        queries = queries[:args.limit]
    run_id = f"{settings.retrieval_backend}-{args.execution_mode}-{args.scope}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    output = args.output_dir / run_id
    print(f"Output: {output}", flush=True)
    print("Initializing models once; startup is excluded from query latency...", flush=True)
    start = perf_counter_ns()
    precompute_ms = None
    if args.scope == "full-query":
        pipeline = create_pipeline(args.device, settings=settings, execution_mode=args.execution_mode)
        lifecycle.callback(pipeline.close)
        runtime_init_ms = (perf_counter_ns() - start) / 1_000_000

        def measure_query(query, timings):
            return pipeline.retrieve(
                query.query, retriever_top_k=args.retriever_top_k,
                coarse_top_k=args.coarse_top_k, fine_top_k=args.fine_top_k,
                embedding_batch_size=args.embedding_batch_size,
                reranker_batch_size=args.reranker_batch_size, timings=timings,
            )
    else:
        store = create_retrieval_store(settings)
        store.require_connection()
        if not store.has_index():
            raise RuntimeError(f"Missing ingested resource: {store.resource_name}")
        embedder = Qwen3Embedder(
            settings.embedding_model_path, device=args.device,
            output_dims=settings.embedding_dims, instruction=settings.retrieval_instruction,
        )
        runtime_init_ms = (perf_counter_ns() - start) / 1_000_000
        print(f"Precomputing {len(queries)} query vectors outside measurement...", flush=True)
        start = perf_counter_ns()
        vectors = embedder.encode_queries(
            [query.query for query in queries], batch_size=args.embedding_batch_size
        )
        if len(vectors) != len(queries):
            raise RuntimeError("Unexpected number of query vectors")
        query_vectors = {query.query_id: vector for query, vector in zip(queries, vectors)}
        precompute_ms = (perf_counter_ns() - start) / 1_000_000
        search_executor = (
            lifecycle.enter_context(ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-search"))
            if args.execution_mode == "concurrent" else None
        )

        def measure_query(query, timings):
            with measure_stage(timings, "total_ms"):
                return hybrid_search(
                    store, query.query, query_vectors[query.query_id],
                    retriever_top_k=args.retriever_top_k,
                    coarse_top_k=args.coarse_top_k, timings=timings,
                    execution_mode=args.execution_mode, executor=search_executor,
                )

    # Explicit whitelist: never dump Settings/.env, which contain passwords.
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "backend": settings.retrieval_backend, "resource": settings.resource_name,
        "milvus_database": settings.milvus_database if settings.retrieval_backend == "milvus" else None,
        "scope": args.scope, "device": args.device, "python": platform.python_version(),
        "hostname": platform.node(), "packages": package_versions(),
        "query_source": str(args.queries) if args.queries else "manual smoke queries",
        "embedding_dims": settings.embedding_dims,
        "embedding_model": str(settings.embedding_model_path),
        "reranker_model": str(settings.reranker_model_path) if args.scope == "full-query" else None,
        "retrieval_instruction": settings.retrieval_instruction,
        "retriever_top_k": args.retriever_top_k, "coarse_top_k": args.coarse_top_k,
        "fine_top_k": args.fine_top_k if args.scope == "full-query" else None,
        "embedding_batch_size": args.embedding_batch_size,
        "reranker_batch_size": args.reranker_batch_size if args.scope == "full-query" else None,
        "num_candidates": 100, "rrf_rank_constant": 60,
        "runtime_init_ms": runtime_init_ms, "query_vector_precompute_ms": precompute_ms,
    }
    summary = run_benchmark(
        queries, measure_query, output, warmup=args.warmup,
        repeats=args.repeats, seed=args.seed, metadata=metadata,
        execution_mode=args.execution_mode,
        progress=lambda message: print(message, flush=True),
    )
    print(f"Successful={summary['successful']}; failed={summary['failed']}", flush=True)
    for field, stats in summary["latency_ms"].items():
        print(f"{field}: mean={stats['mean']:.3f} p50={stats['p50']:.3f} p95={stats['p95']:.3f} p99={stats['p99']:.3f}", flush=True)
    print(f"Saved raw.csv, summary.json and queries.jsonl in {output}", flush=True)
    if summary["failed"]:
        raise SystemExit(1)


def main() -> None:
    with ExitStack() as lifecycle:
        _run(parse_args(), lifecycle)


if __name__ == "__main__":
    main()
