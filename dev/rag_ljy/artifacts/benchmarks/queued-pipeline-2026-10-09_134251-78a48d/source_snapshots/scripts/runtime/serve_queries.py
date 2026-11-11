#!/usr/bin/env python3
"""Keep embedding and reranking models resident and serve repeated HTTP queries."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import logging

from _bootstrap import load_project_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto uses DC_BACKEND (default ascend), or an explicit device such as npu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--execution-mode", choices=["sequential", "concurrent"], default="concurrent")
    parser.add_argument("--max-inflight", type=int, default=8,
                        help="maximum accepted requests, including active and queued work")
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--embedding-workers", type=int, default=2,
                        help="maximum outstanding vLLM embedding calls; local models stay single-threaded")
    parser.add_argument("--stage-queue-capacity", type=int,
                        help="waiting slots per stage; defaults to max-inflight")
    args = parser.parse_args()
    if min(args.max_inflight, args.retrieval_workers, args.embedding_workers) < 1 or (
            args.stage_queue_capacity is not None and args.stage_queue_capacity < 1):
        parser.error("max-inflight and retrieval-workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_project_env()

    import uvicorn
    from rag_engine.server import create_app

    # One process owns one pair of models. No reload or worker recycling.
    uvicorn.run(create_app(args.device, execution_mode=args.execution_mode,
                           max_inflight=args.max_inflight, retrieval_workers=args.retrieval_workers,
                           embedding_workers=args.embedding_workers, stage_queue_capacity=args.stage_queue_capacity), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
