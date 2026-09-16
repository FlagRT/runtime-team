"""Host-only fixture. Never represents a hardware communication result."""
import argparse
import json
import os
from pathlib import Path
import signal
import resource
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("--iterations", type=int)
parser.add_argument("--out-dir", type=Path)
args = parser.parse_args()
if args.mode == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
for rank in range(2):
    row = {
        "schema_version": 2, "run_id": os.environ["COMM_RUN_ID"],
        "rank": rank, "world_size": 2, "iterations": args.iterations,
        "counting_unit": "per_rank_operation_check",
        "dtypes": ["float32", "bfloat16"],
        "calls_per_operation": dict.fromkeys(
            ["all_reduce", "all_gather", "p2p", "async_all_reduce"], args.iterations * 2),
        "total_calls": args.iterations * 8, "passed_calls": args.iterations * 8,
        "failed_calls": 0, "failures": [], "verdict": "PASS",
        "phase": "group_destroyed", "test_fixture": True,
    }
    if args.mode == "missing" and rank == 1:
        continue
    if args.mode == "stale":
        row["run_id"] = "old-run"
    if args.mode == "wrong-rank":
        row["rank"] = 0
    if args.mode == "bad-count":
        row["passed_calls"] -= 1
    if args.mode == "before-destroy":
        row["phase"] = "checks_complete"
    if args.mode == "numeric-fail":
        row.update(verdict="FAIL", failed_calls=1, failures=[{"operation": "p2p"}])
    path = args.out_dir / f"communication_correctness_rank{rank}.json"
    path.write_text("{" if args.mode == "malformed" else json.dumps(row))
if args.mode == "signal":
    os.kill(os.getpid(), signal.SIGTERM)
if args.mode == "abort":
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.abort()
sys.exit(7 if args.mode == "nonzero" else 0)
