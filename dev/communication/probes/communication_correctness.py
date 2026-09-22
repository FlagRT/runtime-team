#!/usr/bin/env python3
"""Deterministic 2-rank FlagCX correctness and completion-semantics probe.

Run inside the locked training image:

  TORCH_DEVICE_BACKEND_AUTOLOAD=0 torchrun --nproc_per_node=2 \
    communication_correctness.py --iterations 20 --out-dir /tmp/comm-results

The public torch.distributed backend name in that image is ``flagos``; its
communication implementation is FlagCX. Each rank writes one JSON result.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch_fl  # noqa: F401 -- locked image requires Torch-FL registration
import torch
import torch.distributed as dist


DTYPES = (torch.float32, torch.bfloat16)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _sync() -> None:
    torch.flagos.synchronize()


def _all_reduce(rank: int, world: int, device: str, dtype: torch.dtype) -> bool:
    value = torch.full((8,), float(rank + 1), dtype=dtype, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    _sync()
    expected = float(world * (world + 1) // 2)
    return bool(torch.all(value.cpu() == expected).item())


def _all_gather(rank: int, world: int, device: str, dtype: torch.dtype) -> bool:
    value = torch.full((4,), float(rank), dtype=dtype, device=device)
    gathered = [torch.empty_like(value) for _ in range(world)]
    dist.all_gather(gathered, value)
    _sync()
    return all(
        bool(torch.all(t.cpu() == float(expected_rank)).item())
        for expected_rank, t in enumerate(gathered)
    )


def _p2p(rank: int, device: str, dtype: torch.dtype) -> bool:
    base = torch.arange(8, dtype=dtype, device=device)
    received = torch.empty_like(base)
    if rank == 0:
        dist.send(base + 10, dst=1)
        dist.recv(received, src=1)
        expected = base + 20
    else:
        dist.recv(received, src=0)
        dist.send(base + 20, dst=0)
        expected = base + 10
    _sync()
    return bool(torch.equal(received.cpu(), expected.cpu()))


def _async_all_reduce(
    rank: int, world: int, device: str, dtype: torch.dtype
) -> tuple[bool, bool]:
    value = torch.full((8,), float(rank + 1), dtype=dtype, device=device)
    work = dist.all_reduce(value, op=dist.ReduceOp.SUM, async_op=True)
    immediate_completed = bool(work.is_completed())
    work.wait()
    _sync()
    expected = float(world * (world + 1) // 2)
    ok = bool(torch.all(value.cpu() == expected).item())
    return ok, immediate_completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--out-dir", default="/tmp/comm-results")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    started = time.perf_counter()
    dist.init_process_group("flagos", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"this probe requires exactly 2 ranks, got {world}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.flagos.set_device(local_rank)
    device = f"flagos:{local_rank}"

    failures: list[dict[str, object]] = []
    immediate_completed: dict[str, list[bool]] = {
        _dtype_name(dtype): [] for dtype in DTYPES
    }
    calls_per_operation = {
        name: 0
        for name in ("all_reduce", "all_gather", "p2p", "async_all_reduce")
    }

    for iteration in range(args.iterations):
        for dtype in DTYPES:
            dtype_name = _dtype_name(dtype)
            checks = {
                "all_reduce": _all_reduce(rank, world, device, dtype),
                "all_gather": _all_gather(rank, world, device, dtype),
                "p2p": _p2p(rank, device, dtype),
            }
            async_ok, completed = _async_all_reduce(rank, world, device, dtype)
            checks["async_all_reduce"] = async_ok
            immediate_completed[dtype_name].append(completed)
            for operation, ok in checks.items():
                calls_per_operation[operation] += 1
                if not ok:
                    failures.append(
                        {"iteration": iteration, "dtype": dtype_name, "operation": operation}
                    )

    elapsed = time.perf_counter() - started
    total_calls = sum(calls_per_operation.values())
    result = {
        "schema_version": 1,
        "backend": "flagos (FlagCX implementation in locked training image)",
        "rank": rank,
        "world_size": world,
        "device": device,
        "torch_version": torch.__version__,
        "iterations": args.iterations,
        "dtypes": [_dtype_name(dtype) for dtype in DTYPES],
        "calls_per_operation": calls_per_operation,
        "total_calls": total_calls,
        "passed_calls": total_calls - len(failures),
        "failed_calls": len(failures),
        "immediate_is_completed_true": {
            dtype: sum(values) for dtype, values in immediate_completed.items()
        },
        "elapsed_s": round(elapsed, 3),
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"communication_correctness_rank{rank}.json"
    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        f"[rank{rank}] {result['verdict']} "
        f"{result['passed_calls']}/{result['total_calls']} calls "
        f"in {result['elapsed_s']}s -> {out_file}",
        flush=True,
    )
    dist.destroy_process_group()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
