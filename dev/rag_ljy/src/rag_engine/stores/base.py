"""Common contract implemented by every retrieval database backend."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol


class RetrievalStore(Protocol):
    """Sparse+dense retrieval operations required by the RAG pipeline."""

    backend_name: str

    @property
    def resource_name(self) -> str: ...

    def require_connection(self) -> None: ...

    def has_index(self) -> bool: ...

    def create_index(self, recreate: bool = False) -> bool: ...

    def bulk_index(
        self, chunks: Iterable[dict[str, Any]]
    ) -> tuple[int, list[Any]]: ...

    def bm25_search(
        self, query: str, size: int = 50
    ) -> list[dict[str, Any]]: ...

    def dense_search(
        self,
        query_vector: list[float],
        size: int = 50,
        num_candidates: int = 100,
    ) -> list[dict[str, Any]]: ...
