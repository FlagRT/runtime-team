"""Construct the configured retrieval database backend."""

from __future__ import annotations

from ..config import Settings
from .base import RetrievalStore


def create_retrieval_store(settings: Settings) -> RetrievalStore:
    if settings.retrieval_backend == "elasticsearch":
        from .elasticsearch import ElasticsearchStore

        return ElasticsearchStore(settings)
    if settings.retrieval_backend == "milvus":
        from .milvus import MilvusStore

        return MilvusStore(settings)
    raise ValueError(f"Unsupported retrieval backend: {settings.retrieval_backend}")
