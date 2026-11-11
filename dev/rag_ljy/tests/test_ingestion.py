from copy import deepcopy

import pytest

from rag_engine.ingestion import ingest_chunks


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = []

    def encode_documents(self, documents, batch_size=8):
        self.calls.append((list(documents), batch_size))
        return [[1.0, float(len(text))] for text in documents]


class FakeStore:
    def __init__(self, backend_name, accepted=None, errors=None) -> None:
        self.backend_name = backend_name
        self.resource_name = "test-resource"
        self.accepted = accepted
        self.errors = errors or []
        self.batches = []

    def bulk_index(self, chunks):
        self.batches.append(deepcopy(chunks))
        success = len(chunks) if self.accepted is None else self.accepted
        return success, self.errors


def make_chunks():
    return [{"chunk_id": str(i), "text": f"text-{i}"} for i in range(5)]


def test_dual_ingestion_embeds_once_and_writes_identical_bounded_batches() -> None:
    chunks = make_chunks()
    embedder = FakeEmbedder()
    es, milvus = FakeStore("elasticsearch"), FakeStore("milvus")

    result = ingest_chunks(
        chunks, [es, milvus], embedder, embedding_dims=2,
        batch_size=1, write_batch_size=2, progress=lambda _: None,
    )

    assert result.indexed_counts == {"elasticsearch": 5, "milvus": 5}
    assert result.probe_vector == [1.0, 6.0]
    assert [len(batch) for batch in es.batches] == [2, 2, 1]
    assert es.batches == milvus.batches
    assert [text for texts, _ in embedder.calls for text in texts] == [
        chunk["text"] for chunk in chunks
    ]
    assert all(batch_size == 1 for _, batch_size in embedder.calls)
    assert all("embedding" not in chunk for chunk in chunks)


@pytest.mark.parametrize("accepted,errors", [(0, []), (2, ["rejected"])])
def test_dual_ingestion_stops_on_partial_or_failed_write(accepted, errors) -> None:
    es = FakeStore("elasticsearch", accepted=accepted, errors=errors)
    milvus = FakeStore("milvus")
    with pytest.raises(RuntimeError, match="rerun the same input"):
        ingest_chunks(
            make_chunks(), [es, milvus], FakeEmbedder(), embedding_dims=2,
            write_batch_size=2, progress=lambda _: None,
        )
    assert not milvus.batches


@pytest.mark.parametrize("vectors", [[], [[1.0]], [[0.0, 0.0]], [[float("nan"), 1.0]]])
def test_invalid_embeddings_are_rejected_before_any_write(vectors) -> None:
    class InvalidEmbedder:
        def encode_documents(self, *_args, **_kwargs):
            return vectors

    store = FakeStore("elasticsearch")
    with pytest.raises((ValueError, RuntimeError)):
        ingest_chunks(
            make_chunks()[:1], [store], InvalidEmbedder(), embedding_dims=2,
            progress=lambda _: None,
        )
    assert not store.batches


def test_duplicate_chunk_ids_are_rejected_before_embedding() -> None:
    embedder = FakeEmbedder()
    chunks = [make_chunks()[0]] * 2
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        ingest_chunks(chunks, [FakeStore("elasticsearch")], embedder, embedding_dims=2)
    assert not embedder.calls
