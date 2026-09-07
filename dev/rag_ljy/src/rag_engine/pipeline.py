"""End-to-end retrieval and reranking pipeline; generation is added later."""

from __future__ import annotations

from typing import Any

from .embedding import Qwen3Embedder
from .reranker import Qwen3Reranker
from .retrieval import hybrid_search
from .stores.base import RetrievalStore


class RetrievalPipeline:
    def __init__(
        self,
        store: RetrievalStore,
        embedder: Qwen3Embedder,
        reranker: Qwen3Reranker,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        retriever_top_k: int = 50,
        coarse_top_k: int = 30,
        fine_top_k: int = 5,
        embedding_batch_size: int = 8,
        reranker_batch_size: int = 4,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedder.encode_queries(
            [query], batch_size=embedding_batch_size
        )[0]
        candidates = hybrid_search(
            self.store,
            query,
            query_vector,
            retriever_top_k=retriever_top_k,
            coarse_top_k=coarse_top_k,
        )
        return self.reranker.rerank(
            query,
            candidates,
            top_k=fine_top_k,
            batch_size=reranker_batch_size,
        )
