"""Document loading and deterministic character-based chunking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            for required in ("document_id", "title", "text", "source_uri"):
                if not value.get(required):
                    raise ValueError(f"{path}:{line_number}: missing {required}")
            documents.append(value)
    return documents


def chunk_documents(
    documents: list[dict[str, Any]],
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars - 1")

    chunks: list[dict[str, Any]] = []
    for document in documents:
        version = int(document.get("document_version", 1))
        for index, text in enumerate(
            split_text(document["text"], max_chars, overlap_chars)
        ):
            identity = (
                f"{document['document_id']}:{version}:{index}:{text}".encode("utf-8")
            )
            chunk_id = hashlib.sha256(identity).hexdigest()
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document["document_id"],
                    "document_version": version,
                    "chunk_index": index,
                    "title": document["title"],
                    "text": text,
                    "source_uri": document["source_uri"],
                    "metadata": document.get("metadata", {}),
                }
            )
    return chunks


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            minimum_break = start + max_chars // 2
            break_at = max(
                normalized.rfind(separator, minimum_break, end)
                for separator in ("。", "！", "？", ". ", "; ", " ")
            )
            if break_at >= minimum_break:
                end = break_at + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks
