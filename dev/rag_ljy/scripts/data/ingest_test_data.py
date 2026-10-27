#!/usr/bin/env python3
"""Chunk, optionally embed, and index the bundled smoke-test documents."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse

from _bootstrap import PROJECT_ROOT, load_project_env

load_project_env()

from rag_engine import Settings, create_retrieval_store  # noqa: E402
from rag_engine.documents import chunk_documents, load_jsonl  # noqa: E402
from rag_engine.vllm_embedding import create_embedder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "sample_documents.jsonl"),
    )
    parser.add_argument("--device", default="auto", help="auto uses DC_BACKEND (default ascend), or an explicit device such as npu:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=150)
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="index text only for the initial BM25 smoke test",
    )
    return parser.parse_args()


def main() -> None:
    from pathlib import Path

    args = parse_args()
    settings = Settings.from_env()
    store = create_retrieval_store(settings)
    store.require_connection()
    if not store.has_index():
        raise RuntimeError(
            f"{store.backend_name} resource {store.resource_name} does not exist; "
            "create it first"
        )

    documents = load_jsonl(Path(args.input))
    chunks = chunk_documents(
        documents,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    if not args.skip_embedding:
        embedder = create_embedder(
            settings.embedding_model_path,
            device=args.device,
            output_dims=settings.embedding_dims,
            instruction=settings.retrieval_instruction,
        )
        vectors = embedder.encode_documents(
            [chunk["text"] for chunk in chunks],
            batch_size=args.batch_size,
        )
        for chunk, vector in zip(chunks, vectors):
            chunk["embedding"] = vector

    success, errors = store.bulk_index(chunks)
    print(
        f"Indexed {success}/{len(chunks)} chunks into "
        f"{store.backend_name}:{store.resource_name}; "
        f"embedding={'no' if args.skip_embedding else 'yes'}"
    )
    if errors:
        raise RuntimeError(
            f"{store.backend_name} rejected {len(errors)} chunks: {errors[:1]}"
        )


if __name__ == "__main__":
    main()
