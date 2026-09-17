#!/usr/bin/env python3
"""Run a sparse BM25 query against the RAG index."""

from __future__ import annotations

import argparse

from _bootstrap import load_project_env

load_project_env()

from rag_engine import Settings, create_retrieval_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        nargs="?",
        default="How are sparse and dense retrieval combined?",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    store = create_retrieval_store(Settings.from_env())
    store.require_connection()
    hits = store.bm25_search(args.query, size=args.top_k)
    print(
        f"backend={store.backend_name} query={args.query!r} hits={len(hits)}"
    )
    for rank, hit in enumerate(hits, start=1):
        source = hit["_source"]
        print(
            f"{rank}. score={hit['_score']:.4f} "
            f"document={source['document_id']} title={source['title']!r} "
            f"source={source['source_uri']}"
        )


if __name__ == "__main__":
    main()
