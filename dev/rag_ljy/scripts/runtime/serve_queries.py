#!/usr/bin/env python3
"""Keep embedding and reranking models resident and serve repeated HTTP queries."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import logging
import math
from dataclasses import replace
from functools import partial
from pathlib import Path

from _bootstrap import load_project_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto uses DC_BACKEND (default ascend), or an explicit device such as npu:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--execution-mode", choices=["sequential", "concurrent"], default="concurrent")
    parser.add_argument("--max-inflight", type=int, default=8,
                        help="maximum accepted requests; 0 disables count rejection (requires --memory-admission)")
    parser.add_argument("--retrieval-workers", type=int, default=4)
    parser.add_argument("--max-reranker-batch-size", type=int, default=4,
                        help="allowed per-request real reranker forward batch (1..50); larger values require measured memory calibration")
    parser.add_argument("--embedding-workers", type=int, default=2,
                        help="maximum outstanding vLLM embedding calls; local models stay single-threaded")
    parser.add_argument("--stage-queue-capacity", type=int,
                        help="waiting slots per stage; defaults to max-inflight, or 8 when count rejection is disabled")
    parser.add_argument("--memory-probe", action=argparse.BooleanOptionalAction, default=True,
                        help="sample accelerator memory through device-context (default enabled)")
    parser.add_argument("--memory-probe-interval", type=float, default=1.0,
                        help="seconds between memory samples (default 1)")
    parser.add_argument("--memory-admission", action="store_true",
                        help="reject new requests when memory or whole-request reservation budgets are insufficient")
    parser.add_argument("--memory-request-mib", type=float,
                        help="legacy fixed-mode budget only; dynamic mode estimates each request")
    parser.add_argument("--memory-budget-mode", choices=['dynamic', 'fixed'], default='dynamic',
                        help="dynamic (default) estimates embedding + rerank before admission; fixed is legacy compatibility")
    parser.add_argument("--memory-safety-mib", type=float, default=5120)
    parser.add_argument("--memory-high-watermark", type=float, default=.85)
    parser.add_argument("--memory-low-watermark", type=float, default=.75)
    parser.add_argument("--memory-max-sample-age", type=float, default=3)
    parser.add_argument("--memory-allow-unverified-scope", action="store_true",
                        help="explicitly accept runtime statistics whose device-wide scope was verified outside the API")
    parser.add_argument('--generation', action=argparse.BooleanOptionalAction, default=None,
                        help='load optional OneRec generation; requires dynamic memory admission')
    parser.add_argument('--generation-model', type=Path)
    parser.add_argument('--generation-max-input-tokens', type=int)
    parser.add_argument('--generation-max-new-tokens', type=int)
    args = parser.parse_args()
    if not 1 <= args.max_reranker_batch_size <= 50:
        parser.error("max-reranker-batch-size must be in 1..50")
    if args.max_inflight < 0 or min(args.retrieval_workers, args.embedding_workers) < 1 or (
            args.stage_queue_capacity is not None and args.stage_queue_capacity < 1):
        parser.error("max-inflight must be nonnegative; worker counts and queue capacity must be positive")
    if not math.isfinite(args.memory_probe_interval) or args.memory_probe_interval <= 0:
        parser.error("memory-probe-interval must be finite and positive")
    from rag_engine.admission import MemoryAdmissionConfig

    try:
        if args.memory_budget_mode == 'dynamic' and args.memory_request_mib is not None:
            raise ValueError('Remove --memory-request-mib for dynamic per-request estimation')
        admission = MemoryAdmissionConfig(
            enabled=args.memory_admission, request_budget_mib=args.memory_request_mib,
            estimate_requests=args.memory_budget_mode == 'dynamic',
            safety_margin_mib=args.memory_safety_mib,
            high_watermark=args.memory_high_watermark, low_watermark=args.memory_low_watermark,
            max_sample_age_s=args.memory_max_sample_age,
            allow_unverified_scope=args.memory_allow_unverified_scope)
        if admission.enabled and not args.memory_probe:
            raise ValueError("memory-admission requires memory-probe")
        if args.max_inflight == 0 and not admission.enabled:
            raise ValueError("--max-inflight 0 requires --memory-admission")
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_project_env()
    from rag_engine.config import Settings
    from rag_engine.generation import GenerationLimits
    from rag_engine.query_runtime import create_pipeline
    settings = Settings.from_env()
    changes = {name: value for name, value in {
        'generation_enabled': args.generation,
        'generation_model_path': args.generation_model,
        'generation_max_input_tokens': args.generation_max_input_tokens,
        'generation_max_new_tokens': args.generation_max_new_tokens}.items() if value is not None}
    settings = replace(settings, **changes)
    try:
        GenerationLimits(settings.generation_max_input_tokens, settings.generation_max_new_tokens)
        if settings.generation_enabled and not (admission.enabled and admission.estimate_requests):
            raise ValueError('--generation requires --memory-admission --memory-budget-mode dynamic')
    except ValueError as error:
        parser.error(str(error))

    import uvicorn
    from rag_engine.server import create_app

    # One process owns one pair of models. No reload or worker recycling.
    factory = partial(create_pipeline, settings=settings, execution_mode=args.execution_mode,
                      safety_mib=admission.safety_margin_mib)
    uvicorn.run(create_app(args.device, pipeline_factory=factory, execution_mode=args.execution_mode,
                           max_inflight=args.max_inflight, retrieval_workers=args.retrieval_workers,
                           embedding_workers=args.embedding_workers, stage_queue_capacity=args.stage_queue_capacity,
                           memory_probe_enabled=args.memory_probe, memory_probe_interval_s=args.memory_probe_interval,
                           memory_admission=admission,
                           max_reranker_batch_size=args.max_reranker_batch_size), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
