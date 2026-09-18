#!/usr/bin/env python3
"""Embed a document JSONL once and index identical chunks in both databases."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from _bootstrap import load_project_env

load_project_env()

from rag_engine import Settings, create_retrieval_store  # noqa: E402
from rag_engine.documents import chunk_documents, load_jsonl  # noqa: E402
from rag_engine.embedding import Qwen3Embedder  # noqa: E402
from rag_engine.ingestion import ingest_chunks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", required=True, help="allocated NPU, e.g. npu:3")
    parser.add_argument("--es-index", default="nfcorpus-chunks-v1")
    parser.add_argument("--milvus-collection", default="nfcorpus_chunks_v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--write-batch-size", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-chars", type=int, default=150)
    args = parser.parse_args()
    if args.batch_size < 1 or args.write_batch_size < 1:
        parser.error("batch sizes must be positive")
    return args


def main() -> None:
    args = parse_args()
    documents = load_jsonl(args.input)
    if len({doc["document_id"] for doc in documents}) != len(documents):
        raise ValueError("Input contains duplicate document IDs")
    chunks = chunk_documents(
        documents, max_chars=args.max_chars, overlap_chars=args.overlap_chars
    )
    if not chunks:
        raise ValueError("Input has no indexable chunks")
    settings = Settings.from_env()
    if not settings.elasticsearch_password:
        raise RuntimeError("Both-backend ingestion requires ELASTIC_PASSWORD")
    stores = [
        create_retrieval_store(replace(
            settings, retrieval_backend="elasticsearch",
            elasticsearch_index_name=args.es_index,
        )),
        create_retrieval_store(replace(
            settings, retrieval_backend="milvus",
            milvus_collection_name=args.milvus_collection,
        )),
    ]
    # Check both databases before creating resources or loading NPU weights.
    for store in stores:
        store.require_connection()
    for store in stores:
        created = store.create_index()  # Never --recreate/drop/delete.
        print(f"{'Created' if created else 'Reusing'} {store.backend_name}:{store.resource_name}", flush=True)
    print(f"Documents={len(documents)}; chunks={len(chunks)}; device={args.device}", flush=True)
    embedder = Qwen3Embedder(
        settings.embedding_model_path, device=args.device,
        output_dims=settings.embedding_dims,
        instruction=settings.retrieval_instruction,
    )
    result = ingest_chunks(
        chunks, stores, embedder, embedding_dims=settings.embedding_dims,
        batch_size=args.batch_size, write_batch_size=args.write_batch_size,
        progress=lambda message: print(message, flush=True),
    )

    es, milvus = stores
    es.client.indices.refresh(index=es.resource_name)
    es_count = int(es.client.count(index=es.resource_name)["count"])
    print("Flushing Milvus and verifying counts...", flush=True)
    milvus.client.flush(collection_name=milvus.resource_name, timeout=120)
    response = milvus.client.query(
        collection_name=milvus.resource_name, filter="",
        output_fields=["count(*)"], consistency_level="Strong", timeout=120,
    )
    milvus_count = int(response[0]["count(*)"])
    for store, count in ((es, es_count), (milvus, milvus_count)):
        print(f"Verified {store.backend_name}:{store.resource_name}: {count} chunks", flush=True)
        if count != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} chunks, found {count} in {store.resource_name}. "
                "The resource may contain older chunks; use a new resource name "
                "rather than deleting existing data."
            )

    # Connectivity/index smoke check, not a relevance-quality benchmark.
    for store in stores:
        if not store.bm25_search(chunks[0]["title"], size=3):
            raise RuntimeError(f"{store.backend_name}: BM25 smoke check returned no hits")
        if not store.dense_search(result.probe_vector, size=3):
            raise RuntimeError(f"{store.backend_name}: dense smoke check returned no hits")
        print(f"{store.backend_name}: BM25 + dense smoke checks passed", flush=True)
    print("DONE: both databases hold the same corpus chunks and embeddings.", flush=True)


if __name__ == "__main__":
    main()
