#!/usr/bin/env python3
"""Run BM25+dense retrieval, RRF coarse ranking, and Qwen3 fine ranking."""

from __future__ import annotations

import argparse
import json

from _bootstrap import load_project_env

load_project_env()

from rag_engine import Settings, create_retrieval_store  # noqa: E402
from rag_engine.embedding import Qwen3Embedder  # noqa: E402
from rag_engine.pipeline import RetrievalPipeline  # noqa: E402
from rag_engine.reranker import Qwen3Reranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--retriever-top-k", type=int, default=50)
    parser.add_argument("--coarse-top-k", type=int, choices=range(20, 51), default=30)
    parser.add_argument("--fine-top-k", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    store = create_retrieval_store(settings)
    store.require_connection()
    embedder = Qwen3Embedder(
        settings.embedding_model_path,
        device=args.device,
        output_dims=settings.embedding_dims,
        instruction=settings.retrieval_instruction,
    )
    reranker = Qwen3Reranker(
        settings.reranker_model_path,
        device=args.device,
        instruction=settings.retrieval_instruction,
    )
    pipeline = RetrievalPipeline(store, embedder, reranker)
    hits = pipeline.retrieve(
        args.query,
        retriever_top_k=args.retriever_top_k,
        coarse_top_k=args.coarse_top_k,
        fine_top_k=args.fine_top_k,
        embedding_batch_size=args.embedding_batch_size,
        reranker_batch_size=args.reranker_batch_size,
    )

    result = [
        {
            "rank": rank,
            "rerank_score": hit["_rerank_score"],
            "rrf_score": hit["_rrf_score"],
            "retrieval_ranks": hit["_retrieval_ranks"],
            "document_id": hit["_source"]["document_id"],
            "chunk_id": hit["_source"]["chunk_id"],
            "title": hit["_source"]["title"],
            "text": hit["_source"]["text"],
            "source_uri": hit["_source"]["source_uri"],
        }
        for rank, hit in enumerate(hits, start=1)
    ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
