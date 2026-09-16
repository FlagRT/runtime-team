#!/usr/bin/env python3
"""POSIX process supervisor: numeric PASS alone is not lifecycle acceptance.

Use inside the locked image, after checking free cards/container limits:
  python run_acceptance.py --run-root /tmp/comm-runs -- \
    torchrun --nproc_per_node=2 communication_correctness.py
No device libraries are imported by this supervisor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import uuid


OPERATIONS = {"all_reduce", "all_gather", "p2p", "async_all_reduce"}


def inspect_results(directory: Path, run_id: str, iterations: int) -> dict:
    errors = []
    numeric_ok = True
    cleanup_ok = True
    for rank in range(2):
        try:
            row = json.loads((directory / f"communication_correctness_rank{rank}.json").read_text())
            expected = iterations * 2
            valid = (
                row["schema_version"] == 2 and row["run_id"] == run_id
                and row["rank"] == rank and row["world_size"] == 2
                and row["iterations"] == iterations
                and row["counting_unit"] == "per_rank_operation_check"
                and row["dtypes"] == ["float32", "bfloat16"]
                and row["calls_per_operation"] == dict.fromkeys(OPERATIONS, expected)
                and row["total_calls"] == expected * 4
                and row["passed_calls"] == expected * 4
                and row["failed_calls"] == 0 and row["failures"] == []
                and row["verdict"] == "PASS"
            )
            if not valid:
                errors.append(f"rank{rank}: inconsistent or unsuccessful checks")
                numeric_ok = False
            if row.get("phase") != "group_destroyed":
                errors.append(f"rank{rank}: process group cleanup not recorded")
                cleanup_ok = False
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"rank{rank}: {type(exc).__name__}: {exc}")
            numeric_ok = cleanup_ok = False
    return {"numeric_ok": numeric_ok, "cleanup_ok": cleanup_ok, "errors": errors}


def signal_group(pid: int, sig: int) -> bool:
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False


def run(command: list[str], root: Path, iterations: int = 20,
        timeout: float = 240, grace: float = 3) -> dict:
    if not command or iterations < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("command, positive iterations and finite positive timeout required")
    if not math.isfinite(grace) or grace <= 0:
        raise ValueError("grace must be finite and positive")
    run_id = uuid.uuid4().hex
    directory = root.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, COMM_RUN_ID=run_id, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    argv = [*command, "--iterations", str(iterations), "--out-dir", str(directory)]
    timed_out = False
    forced_cleanup = False
    returncode = None
    launch_error = None
    with (directory / "launcher.log").open("w") as log:
        try:
            process = subprocess.Popen(argv, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            launch_error = str(exc)
        else:
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                signal_group(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    signal_group(process.pid, signal.SIGKILL)
                    returncode = process.wait()
            finally:
                # Also catch workers orphaned by a launcher that exited first.
                forced_cleanup = signal_group(process.pid, signal.SIGKILL)
                if process.returncode is None:
                    process.wait()
    report = inspect_results(directory, run_id, iterations)
    report.update(run_id=run_id, output_dir=str(directory), command=argv,
                  returncode=returncode, timed_out=timed_out,
                  forced_cleanup=forced_cleanup, launch_error=launch_error,
                  validation_scope="process_and_result_contract")
    report["verdict"] = "PASS" if (
        report["numeric_ok"] and report["cleanup_ok"] and returncode == 0
        and not timed_out and not forced_cleanup and launch_error is None
    ) else "FAIL"
    (directory / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = run(command, args.run_root, args.iterations, args.timeout)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
