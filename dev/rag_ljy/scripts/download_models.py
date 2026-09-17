#!/usr/bin/env python3
"""Download the two selected Qwen3 retrieval models to the RAID model directory."""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from _bootstrap import load_project_env

load_project_env()

from rag_engine.config import (  # noqa: E402
    DEFAULT_EMBEDDING_REPO,
    DEFAULT_RERANKER_REPO,
    Settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("embedding", "reranker", "all"),
        default="all",
    )
    return parser.parse_args()


def download(repo_id: str, destination) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {destination}")
    snapshot_download(repo_id=repo_id, local_dir=str(destination))


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if args.model in ("embedding", "all"):
        download(DEFAULT_EMBEDDING_REPO, settings.embedding_model_path)
    if args.model in ("reranker", "all"):
        download(DEFAULT_RERANKER_REPO, settings.reranker_model_path)


if __name__ == "__main__":
    main()
