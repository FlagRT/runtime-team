"""Benchmark query loading and BEIR split selection (no model dependencies)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query: str


def load_queries(path: Path) -> list[BenchmarkQuery]:
    """Read prepared query_id/query or BEIR _id/text JSONL."""
    queries = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            identifier = record.get("query_id", record.get("_id"))
            text = record.get("query", record.get("text"))
            if identifier is None or not str(identifier).strip():
                raise ValueError(f"{path}:{number}: missing query ID")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{number}: missing query text")
            identifier = str(identifier)
            if identifier in seen:
                raise ValueError(f"{path}:{number}: duplicate query ID {identifier}")
            seen.add(identifier)
            queries.append(BenchmarkQuery(identifier, text))
    if not queries:
        raise ValueError(f"No queries in {path}")
    return queries


def select_qrels_queries(
    queries: list[BenchmarkQuery], qrels: Path
) -> list[BenchmarkQuery]:
    with qrels.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "query-id" not in reader.fieldnames:
            raise ValueError(f"{qrels}: missing query-id TSV column")
        identifiers = {row["query-id"] for row in reader if row.get("query-id")}
    missing = identifiers - {query.query_id for query in queries}
    if missing:
        raise ValueError(f"Queries missing from input: {sorted(missing)[:5]}")
    selected = sorted(
        (query for query in queries if query.query_id in identifiers),
        key=lambda query: query.query_id,
    )
    if not selected:
        raise ValueError(f"No queries selected by {qrels}")
    return selected
