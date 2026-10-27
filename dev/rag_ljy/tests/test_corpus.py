import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_engine.corpus import load_corpus, normalize_corpus_record
from rag_engine.documents import chunk_documents


def test_corpus_preserves_ids_and_supplies_missing_title() -> None:
    record = {"_id": "MED-10", "text": "A medical document.", "title": ""}
    document = normalize_corpus_record(record, "nfcorpus")

    assert document["document_id"] == "MED-10"
    assert document["title"] == "MED-10"
    assert document["source_uri"] == "beir://nfcorpus/MED-10"
    assert document["metadata"] == {"dataset": "nfcorpus"}
    assert chunk_documents([document])[0]["document_id"] == "MED-10"


@pytest.mark.parametrize("record", [
    {"text": "Missing ID"},
    {"_id": "MED-1", "text": " "},
    {"_id": "MED-1", "text": "Text", "title": 42},
])
def test_corpus_rejects_invalid_records(record) -> None:
    with pytest.raises(ValueError):
        normalize_corpus_record(record, "nfcorpus")


def test_corpus_jsonl_rejects_duplicate_document_ids(tmp_path) -> None:
    path = tmp_path / "corpus.jsonl"
    record = {"_id": "MED-1", "title": "Title", "text": "Text"}
    path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_corpus(path, "nfcorpus")


def test_corpus_parquet_reads_only_corpus_columns(monkeypatch, tmp_path) -> None:
    shard = tmp_path / "corpus-00000.parquet"
    shard.touch()
    record = {"_id": "MED-1", "title": "Title", "text": "Text"}

    class FakeReader:
        schema_arrow = SimpleNamespace(names=["_id", "title", "text"])

        def __init__(self, source) -> None:
            assert source == shard

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def iter_batches(self, batch_size, columns):
            assert batch_size == 512
            assert columns == ["_id", "title", "text"]
            yield SimpleNamespace(to_pylist=lambda: [record])

    arrow = ModuleType("pyarrow")
    parquet = ModuleType("pyarrow.parquet")
    parquet.ParquetFile = FakeReader
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)

    assert load_corpus(tmp_path, "nfcorpus")[0]["document_id"] == "MED-1"


def test_queries_parquet_is_rejected(monkeypatch, tmp_path) -> None:
    path = tmp_path / "queries.parquet"
    path.touch()

    class FakeReader:
        schema_arrow = SimpleNamespace(names=["_id", "text"])

        def __init__(self, source):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    arrow = ModuleType("pyarrow")
    parquet = ModuleType("pyarrow.parquet")
    parquet.ParquetFile = FakeReader
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    with pytest.raises(ValueError, match="expected corpus columns"):
        load_corpus(path, "nfcorpus")
