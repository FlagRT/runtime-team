"""Bounded dual-backend ingestion: embed once, reuse identical chunk vectors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .stores.base import RetrievalStore


class DocumentEmbedder(Protocol):
    def encode_documents(
        self, documents: list[str], batch_size: int = 8
    ) -> list[list[float]]: ...


@dataclass(frozen=True)
class IngestionResult:
    indexed_counts: dict[str, int]
    probe_vector: list[float]


def ingest_chunks(
    chunks: list[dict[str, Any]],
    stores: Sequence[RetrievalStore],
    embedder: DocumentEmbedder,
    *,
    embedding_dims: int,
    batch_size: int = 8,
    write_batch_size: int = 64,
    progress: Callable[[str], None] = print,
) -> IngestionResult:
    """Upsert batches to every store, aborting on any rejected or partial batch.

    The two databases are not one transaction. If a later write fails, earlier
    successful writes remain; rerun the same input/settings to safely upsert
    the same deterministic chunk IDs. No index or collection is ever deleted.
    """
    if batch_size < 1 or write_batch_size < 1 or embedding_dims < 1:
        raise ValueError("Batch sizes and embedding_dims must be positive")
    if not chunks or not stores:
        raise ValueError("Ingestion requires non-empty chunks and stores")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Input contains duplicate chunk IDs")
    totals = {store.backend_name: 0 for store in stores}
    probe_vector: list[float] = []
    for start in range(0, len(chunks), write_batch_size):
        # Keep embeddings bounded to the current write batch, not the full corpus.
        batch = [dict(chunk) for chunk in chunks[start:start + write_batch_size]]
        progress(f"Embedding chunks {start + 1}-{start + len(batch)}/{len(chunks)}")
        vectors = embedder.encode_documents(
            [chunk["text"] for chunk in batch], batch_size=batch_size
        )
        if len(vectors) != len(batch):
            raise RuntimeError("Embedder returned an unexpected number of vectors")
        for chunk, vector in zip(batch, vectors):
            if (
                len(vector) != embedding_dims
                or not all(math.isfinite(value) for value in vector)
                or not any(value != 0 for value in vector)
            ):
                raise ValueError(f"Invalid embedding for chunk {chunk['chunk_id']}")
            chunk["embedding"] = vector
        if start == 0:
            probe_vector = list(vectors[0])
        for store in stores:
            success, errors = store.bulk_index(batch)
            if errors or success != len(batch):
                raise RuntimeError(
                    f"{store.backend_name}:{store.resource_name} accepted "
                    f"{success}/{len(batch)} chunks; rejected={len(errors)}. "
                    "Earlier batches may remain; rerun the same input to upsert."
                )
            totals[store.backend_name] += success
            progress(f"  {store.backend_name}: {totals[store.backend_name]}/{len(chunks)} written")
    return IngestionResult(indexed_counts=totals, probe_vector=probe_vector)
