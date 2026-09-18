#!/usr/bin/env python3
"""Host-side BEIR query Parquet conversion, selecting the requested qrels split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_engine.benchmark_data import (  # noqa: E402
    BenchmarkQuery, load_queries, select_qrels_queries,
)


def read_queries(path: Path) -> list[BenchmarkQuery]:
    if path.suffix == ".jsonl":
        return load_queries(path)
    import pyarrow.parquet as parquet

    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"No query Parquet shards in {path}")
    queries = []
    seen = set()
    for source in files:
        with parquet.ParquetFile(source) as reader:
            if not {"_id", "text"}.issubset(reader.schema_arrow.names):
                raise ValueError(f"{source}: missing query columns _id/text")
            for batch in reader.iter_batches(batch_size=512, columns=["_id", "text"]):
                for record in batch.to_pylist():
                    identifier, text = record["_id"], record["text"]
                    if identifier is None or not str(identifier).strip():
                        raise ValueError("Missing query ID")
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f"Missing text for query {identifier}")
                    identifier = str(identifier)
                    if identifier in seen:
                        raise ValueError(f"Duplicate query ID: {identifier}")
                    seen.add(identifier)
                    queries.append(BenchmarkQuery(identifier, text))
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="BEIR queries/ Parquet directory or queries.jsonl")
    parser.add_argument("--qrels", type=Path, required=True, help="split TSV, e.g. test.tsv")
    parser.add_argument("--output", type=Path, help="new JSONL; omit for a read-only preview")
    args = parser.parse_args()
    queries = select_qrels_queries(read_queries(args.queries), args.qrels)
    print(f"Selected {len(queries)} queries from {args.qrels.name}")
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            for query in queries:
                handle.write(json.dumps(
                    {"query_id": query.query_id, "query": query.query},
                    ensure_ascii=False,
                ) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
