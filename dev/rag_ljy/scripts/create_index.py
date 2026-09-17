#!/usr/bin/env python3
"""Create the configured retrieval index or collection."""

from __future__ import annotations

import argparse

from _bootstrap import load_project_env

load_project_env()

from rag_engine import Settings, create_retrieval_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete and recreate the index, destroying its current documents",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    store = create_retrieval_store(settings)
    store.require_connection()
    created = store.create_index(recreate=args.recreate)
    action = "created" if created else "already exists"
    print(
        f"{store.backend_name} resource {store.resource_name} {action} "
        f"(embedding dims={settings.embedding_dims})"
    )


if __name__ == "__main__":
    main()
