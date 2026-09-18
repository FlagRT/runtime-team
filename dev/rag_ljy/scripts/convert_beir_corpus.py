#!/usr/bin/env python3
"""Convert BEIR corpus Parquet/JSONL into the project's document JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This host-side tool deliberately needs neither dotenv nor the NPU packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_engine.corpus import load_corpus  # noqa: E402
from rag_engine.documents import chunk_documents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="corpus/ or corpus.jsonl")
    parser.add_argument("--dataset-name", default="nfcorpus")
    parser.add_argument(
        "--output", type=Path,
        help="new document JSONL file; omit to preview without writing anything",
    )
    args = parser.parse_args()
    documents = load_corpus(args.input, args.dataset_name)
    chunks = chunk_documents(documents)
    print(f"Documents={len(documents)}; chunks={len(chunks)} (1200 chars, overlap 150)")
    if args.output is not None:
        # Never silently overwrite an existing data artifact.
        with args.output.open("x", encoding="utf-8") as handle:
            for document in documents:
                handle.write(json.dumps(document, ensure_ascii=False) + "\n")
        print(f"Wrote {args.output}; original document IDs preserved")


if __name__ == "__main__":
    main()
