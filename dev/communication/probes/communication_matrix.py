"""Two-rank hardware matrix with explicit producer/consumer synchronization.

This is a functional test, not a bandwidth benchmark. Retain input buffers
through completion. Use an external process timeout and inspect launcher exit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import timedelta

import torch_fl  # noqa: F401
import torch
import torch.distributed as dist


def identity():
    import flagcx
    import flagcx._C
    group = dist.distributed_c10d._get_default_group()
    inner = getattr(group, "_inner", None)
    paths = {str(Path(flagcx._C.__file__).resolve())}
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "libflagcx.so" in line:
            paths.add(line.split()[-1])
    backend_name = inner.name() if hasattr(inner, "name") else None
    result = {"inner_type": str(type(inner)), "inner_name": backend_name, "libraries": []}
    for path in sorted(paths):
        result["libraries"].append({"path": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    if backend_name != "flagcx":
        raise RuntimeError(f"FlagCX backend not confirmed: {type(inner)}, name={backend_name}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--sizes", type=int, nargs="+", default=[8, 1024, 65536])
    p.add_argument("--cross-stream", action="store_true")
    args = p.parse_args()
    if args.iterations < 1 or not args.sizes or min(args.sizes) < 1:
        p.error("positive iterations and element counts required")
    rank = int(os.environ["LOCAL_RANK"])
    torch.flagos.set_device(rank)
    dist.init_process_group("flagos", timeout=timedelta(seconds=90))
    assert dist.get_world_size() == 2
    device = f"flagos:{rank}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = {"rank": rank, "iterations": args.iterations, "sizes": args.sizes,
              "identity": identity(), "mode": "cross_stream" if args.cross_stream else "explicit_sync",
              "checks": [], "phase": "running", "counting_unit": "per_rank_operation_check"}
    output = args.out_dir / f"matrix_rank{rank}.json"
    output.write_text(json.dumps(result, indent=2))
    sync = torch.flagos.synchronize
    started = time.monotonic()
    for dtype in (torch.float32, torch.bfloat16):
        for size in args.sizes:
            for iteration in range(args.iterations):
                offset = iteration % 17

                def record(op, actual, expected):
                    actual_cpu = actual.cpu().float()
                    expect_cpu = expected.cpu().float()
                    diff = float((actual_cpu - expect_cpu).abs().max().item())
                    ok = bool(torch.equal(actual_cpu, expect_cpu))
                    row = {"operation": op, "dtype": str(dtype), "elements": size,
                           "iteration": iteration, "ok": ok, "max_abs_diff": diff}
                    if not ok:
                        row.update(actual_head=actual_cpu.flatten()[:8].tolist(),
                                   expected_head=expect_cpu.flatten()[:8].tolist())
                    result["checks"].append(row)

                if args.cross_stream:
                    producer, consumer = torch.flagos.Stream(), torch.flagos.Stream()
                    with torch.flagos.stream(producer):
                        value = torch.full((size,), rank + 1 + offset, dtype=dtype, device=device)
                        # Do not globally sync between production, communication and consumption.
                        work = dist.all_reduce(value, async_op=True)
                        work.wait()
                        done = torch.flagos.Event()
                        done.record(producer)
                    with torch.flagos.stream(consumer):
                        consumer.wait_event(done)
                        consumed = value * 2
                    consumer.synchronize()
                    record("async_ar_event_consumer", consumed,
                           torch.full((size,), 2 * (3 + 2 * offset), dtype=dtype))
                    sync()
                    continue

                value = torch.full((size,), rank + 1 + offset, dtype=dtype, device=device)
                sync()
                dist.all_reduce(value)
                sync()
                record("all_reduce", value, torch.full((size,), 3 + 2 * offset, dtype=dtype))

                value = torch.full((size,), rank + offset, dtype=dtype, device=device)
                gathered = [torch.empty_like(value) for _ in range(2)]
                sync()
                dist.all_gather(gathered, value)
                sync()
                record("all_gather", torch.stack(gathered),
                       torch.stack([torch.full((size,), r + offset, dtype=dtype) for r in range(2)]))

                send = torch.full((size,), rank + 10 + offset, dtype=dtype, device=device)
                recv = torch.empty_like(send)
                sync()
                if rank == 0:
                    dist.send(send, 1)
                    dist.recv(recv, 1)
                else:
                    dist.recv(recv, 0)
                    dist.send(send, 0)
                sync()
                record("p2p_retained", recv, torch.full((size,), (1-rank) + 10 + offset, dtype=dtype))

                value = torch.full((size,), rank + 1 + offset, dtype=dtype, device=device)
                sync()
                work = dist.all_reduce(value, async_op=True)
                work.wait()
                sync()
                record("async_all_reduce", value, torch.full((size,), 3 + 2 * offset, dtype=dtype))
    result.update(elapsed_s=round(time.monotonic()-started, 3), phase="checks_complete")
    result["passed"] = sum(r["ok"] for r in result["checks"])
    result["total"] = len(result["checks"])
    output.write_text(json.dumps(result, indent=2))
    dist.destroy_process_group()
    result["phase"] = "group_destroyed"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "identity")}), flush=True)
    raise SystemExit(0 if result["passed"] == result["total"] else 1)


if __name__ == "__main__":
    main()
