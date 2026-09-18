#!/usr/bin/env python3
"""Run BM25+dense retrieval, RRF coarse ranking, and Qwen3 fine ranking."""

from __future__ import annotations

import argparse
import json

from _bootstrap import load_project_env

load_project_env()

from rag_engine.query_runtime import create_pipeline, format_hits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--retriever-top-k", type=int, default=50)
    parser.add_argument("--coarse-top-k", type=int, choices=range(20, 51), default=30)
    parser.add_argument("--fine-top-k", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--timings", action="store_true", help="include per-stage elapsed milliseconds; excludes startup")
    parser.add_argument("--execution-mode", choices=["sequential", "concurrent"], default="concurrent")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = create_pipeline(args.device, execution_mode=args.execution_mode)
    timings = {} if args.timings else None
    try:
        hits = pipeline.retrieve(
            args.query,
            retriever_top_k=args.retriever_top_k,
            coarse_top_k=args.coarse_top_k,
            fine_top_k=args.fine_top_k,
            embedding_batch_size=args.embedding_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            timings=timings,
        )
    finally:
        pipeline.close()

    result = format_hits(hits)
    if args.timings:
        result = {"execution_mode": args.execution_mode, "timings_ms": timings, "results": result}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
