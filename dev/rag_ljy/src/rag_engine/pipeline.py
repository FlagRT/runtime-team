"""End-to-end retrieval and reranking pipeline; generation is added later."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .embedding import Qwen3Embedder
from .reranker import Qwen3Reranker
from .retrieval import hybrid_search, validate_execution_mode
from .stores.base import RetrievalStore
from .timing import measure_stage


class RetrievalPipeline:
    def __init__(
        self,
        store: RetrievalStore,
        embedder: Qwen3Embedder,
        reranker: Qwen3Reranker,
        *,
        execution_mode: str = "concurrent",
    ) -> None:
        validate_execution_mode(execution_mode)
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.execution_mode = execution_mode
        self._search_executor = (
            ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-search")
            if execution_mode == "concurrent" else None
        )
        self._closed = False

    def close(self) -> None:
        """Release the owned search pool after in-flight inference has finished."""
        if not self._closed:
            self._closed = True
            if self._search_executor is not None:
                self._search_executor.shutdown(wait=True, cancel_futures=True)

    def retrieve(
        self,
        query: str,
        retriever_top_k: int = 50,
        coarse_top_k: int = 30,
        fine_top_k: int = 5,
        embedding_batch_size: int = 8,
        reranker_batch_size: int = 4,
        timings: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Retrieval pipeline is closed")
        if timings is not None:
            timings.clear()
        with measure_stage(timings, "total_ms"):
            with measure_stage(timings, "embedding_ms"):
                query_vector = self.embedder.encode_queries(
                    [query], batch_size=embedding_batch_size
                )[0]
            candidates = hybrid_search(
                self.store,
                query,
                query_vector,
                retriever_top_k=retriever_top_k,
                coarse_top_k=coarse_top_k,
                timings=timings,
                execution_mode=self.execution_mode,
                executor=self._search_executor,
            )
            with measure_stage(timings, "rerank_ms"):
                return self.reranker.rerank(
                    query,
                    candidates,
                    top_k=fine_top_k,
                    batch_size=reranker_batch_size,
                )
