#!/usr/bin/env python3
"""Task 1 optional follow-up: monkeypatch torch.distributed's broadcast/
all_gather/all_gather_object to call torch.npu.synchronize() immediately
after the real collective, before the caller reads the result -- then run
the real train_qwen_1_5b_npu.py under that patch, to see whether DDP
construction (which internally verifies param shapes across ranks via a
broadcast/all_gather-shaped check) succeeds.

Caveat (know this going in, report honestly either way): PyTorch's
`_verify_param_shape_across_processes` (called from inside
`DistributedDataParallel.__init__`) actually calls the C++-level primitive
`torch._C._distributed_c10d._verify_params_across_processes(...)`, NOT the
Python `torch.distributed.broadcast`/`all_gather` functions patched here. If
that's the actual code path DDP uses, this monkeypatch will NOT intercept
it and DDP construction will fail identically to before -- that is itself a
valid, reportable finding (not a bug in this test), not a sign the sync
hypothesis is false (the direct flagcx_sync_test.py result is the
authoritative test for the hypothesis itself).
"""
import runpy

import torch
import torch.distributed as dist

_orig_broadcast = dist.broadcast
_orig_all_gather = dist.all_gather
_orig_all_gather_object = dist.all_gather_object


def _synced_broadcast(*a, **k):
    r = _orig_broadcast(*a, **k)
    torch.npu.synchronize()
    return r


def _synced_all_gather(*a, **k):
    r = _orig_all_gather(*a, **k)
    torch.npu.synchronize()
    return r


def _synced_all_gather_object(*a, **k):
    r = _orig_all_gather_object(*a, **k)
    torch.npu.synchronize()
    return r


dist.broadcast = _synced_broadcast
dist.all_gather = _synced_all_gather
dist.all_gather_object = _synced_all_gather_object

print("[syncpatch] torch.distributed.broadcast / all_gather / all_gather_object "
      "patched to torch.npu.synchronize() immediately after each call.", flush=True)

runpy.run_path("/workspace/train_qwen_1_5b_npu.py", run_name="__main__")
