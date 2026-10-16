"""Shared model initialization and output for CLI and persistent queries."""

from __future__ import annotations


def create_pipeline(device: str):
    from . import Settings, create_retrieval_store
    from .embedding import Qwen3Embedder
    from .pipeline import RetrievalPipeline
    from .reranker import Qwen3Reranker

    settings = Settings.from_env()
    store = create_retrieval_store(settings)
    store.require_connection()
    if not store.has_index():
        raise RuntimeError(
            f"{store.backend_name} resource {store.resource_name} does not exist; "
            "run scripts/create_index.py and ingest documents first"
        )
    embedder = Qwen3Embedder(
        settings.embedding_model_path,
        device=device,
        output_dims=settings.embedding_dims,
        instruction=settings.retrieval_instruction,
    )
    reranker = Qwen3Reranker(
        settings.reranker_model_path,
        device=device,
        instruction=settings.retrieval_instruction,
    )
    return RetrievalPipeline(store, embedder, reranker)


def format_hits(hits: list[dict]) -> list[dict]:
    return [
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
