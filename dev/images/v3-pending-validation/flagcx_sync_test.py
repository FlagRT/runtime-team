#!/usr/bin/env python3
"""Task 1 (project-lead request): test whether the FlagCX broadcast/all_gather
STOP CONDITION documented in ascend-train-comm/v3/REBUILD.md was actually a
missing torch.npu.synchronize() artifact, per commit 5d545c9's hypothesis
("flagcx backend returns async, must sync before reading result").

Runs on real 2-card hardware via:
  torchrun --nproc_per_node=2 flagcx_sync_test.py

For EACH of broadcast and all_gather (tensor-list form), this runs the SAME
collective twice back-to-back on freshly-reset tensors:
  (a) "no-sync" path: exactly as the original STOP CONDITION repro did (read
      the result tensor immediately after the collective call, no sync) --
      to reconfirm the original finding still reproduces in this new v3 image
      (image id 43f3e2f70b4c, built on the vllm-plugin-FL parent).
  (b) "with-sync" path: torch.npu.synchronize() inserted immediately after
      the collective call and before reading/comparing the result tensor --
      the exact fix commit 5d545c9 applies.

Prints a clear PASS/FAIL verdict for each of the 4 (op x sync-mode)
combinations, from both ranks, so there's no ambiguity about which
mode/collective is at issue.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def log(rank, msg):
    print(f"[rank{rank}] {msg}", flush=True)


def test_broadcast(rank, world_size, device, use_sync):
    label = "broadcast" + ("+sync" if use_sync else "+nosync")
    src_val = 1
    other_val = 2
    if rank == 0:
        t = torch.full((4,), src_val, dtype=torch.int64, device=device)
    else:
        t = torch.full((4,), other_val, dtype=torch.int64, device=device)
    dist.broadcast(t, src=0)
    if use_sync:
        torch.npu.synchronize()
    result = t.cpu().tolist()
    expected = [src_val] * 4
    ok = result == expected
    log(rank, f"{label}: result={result} expected={expected} -> {'PASS' if ok else 'FAIL'}")
    return ok


def test_all_gather(rank, world_size, device, use_sync):
    label = "all_gather" + ("+sync" if use_sync else "+nosync")
    local = torch.full((4,), rank + 1, dtype=torch.int64, device=device)
    out_list = [torch.zeros(4, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(out_list, local)
    if use_sync:
        torch.npu.synchronize()
    result = [o.cpu().tolist() for o in out_list]
    expected = [[r + 1] * 4 for r in range(world_size)]
    ok = result == expected
    log(rank, f"{label}: result={result} expected={expected} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    dist.init_process_group(backend="flagcx")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    device = f"npu:{local_rank}"

    backend_name = torch._C._get_privateuse1_backend_name()
    log(rank, f"PrivateUse1 backend = {backend_name!r}, device={device}, world_size={world_size}")

    results = {}
    # no-sync baseline first (reconfirm original STOP CONDITION reproduces)
    results["broadcast_nosync"] = test_broadcast(rank, world_size, device, use_sync=False)
    results["all_gather_nosync"] = test_all_gather(rank, world_size, device, use_sync=False)
    # with-sync (the fix hypothesis from commit 5d545c9)
    results["broadcast_sync"] = test_broadcast(rank, world_size, device, use_sync=True)
    results["all_gather_sync"] = test_all_gather(rank, world_size, device, use_sync=True)

    dist.barrier()
    log(rank, f"SUMMARY: {results}")

    all_sync_pass = results["broadcast_sync"] and results["all_gather_sync"]
    all_nosync_pass = results["broadcast_nosync"] and results["all_gather_nosync"]
    if rank == 0:
        print("\n=== VERDICT (rank0) ===", flush=True)
        print(f"no-sync path (reconfirm original STOP CONDITION): {'STILL BROKEN' if not all_nosync_pass else 'unexpectedly passed'}", flush=True)
        print(f"with-sync path (5d545c9 hypothesis): {'FIXED by sync' if all_sync_pass else 'STILL BROKEN even with sync'}", flush=True)

    dist.destroy_process_group()
    return 0 if all_sync_pass else 1


if __name__ == "__main__":
    sys.exit(main())
