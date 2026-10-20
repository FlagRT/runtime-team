"""Read BEIR corpus records without changing their document identifiers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import quote


def normalize_corpus_record(
    record: dict[str, Any], dataset_name: str
) -> dict[str, Any]:
    identifier = record.get("_id")
    if identifier is None or not str(identifier).strip():
        raise ValueError("Corpus record is missing _id")
    identifier = str(identifier)
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Corpus document {identifier!r} is missing text")
    title = record.get("title") or identifier
    if not isinstance(title, str):
        raise ValueError(f"Corpus document {identifier!r} has a non-string title")
    return {
        "document_id": identifier,
        "document_version": 1,
        "title": title if title.strip() else identifier,
        "text": text,
        "source_uri": f"beir://{quote(dataset_name, safe='')}/{quote(identifier, safe='')}",
        "metadata": {"dataset": dataset_name},
    }


def iter_corpus_records(path: Path) -> Iterator[dict[str, Any]]:
    """Accept a corpus Parquet shard/directory or BEIR corpus.jsonl.

    Directories must point to corpus/, not the whole dataset: this prevents
    accidentally mixing the queries Parquet files into the indexed corpus.
    """
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise ValueError(f"No Parquet shards in {path}; point to corpus/")
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(path)

    for source in files:
        if source.suffix == ".parquet":
            try:
                import pyarrow.parquet as parquet
            except ImportError as error:
                raise RuntimeError(
                    "Parquet input requires pyarrow; run the converter in your "
                    "host sol environment, not the NPU virtualenv"
                ) from error
            with parquet.ParquetFile(source) as reader:
                if not {"_id", "title", "text"}.issubset(reader.schema_arrow.names):
                    raise ValueError(f"{source}: expected corpus columns _id, title, text")
                for batch in reader.iter_batches(
                    batch_size=512, columns=["_id", "title", "text"]
                ):
                    yield from batch.to_pylist()
        elif source.suffix == ".jsonl":
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{source}:{line_number}: expected a JSON object")
                    yield value
        else:
            raise ValueError(f"Unsupported corpus format: {source}")


def load_corpus(path: Path, dataset_name: str) -> list[dict[str, Any]]:
    documents = []
    seen = set()
    for record in iter_corpus_records(path):
        document = normalize_corpus_record(record, dataset_name)
        identifier = document["document_id"]
        if identifier in seen:
            raise ValueError(f"Duplicate corpus document_id: {identifier}")
        seen.add(identifier)
        documents.append(document)
    if not documents:
        raise ValueError(f"Corpus is empty: {path}")
    return documents
