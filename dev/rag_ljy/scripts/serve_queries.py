#!/usr/bin/env python3
"""Keep embedding and reranking models resident and serve repeated HTTP queries."""

from __future__ import annotations

import argparse

from _bootstrap import load_project_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    args = parser.parse_args()
    load_project_env()

    import uvicorn
    from rag_engine.server import create_app

    # One process owns one pair of models. No reload or worker recycling.
    uvicorn.run(create_app(args.device), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
