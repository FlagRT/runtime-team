"""A 线 flagcxWork 完成语义实测：is_completed / future 行为。

目标：确认 DDP 完成判定依赖的路径（future 谎报 vs wait 语义）。
"""
import os
import torch
import torch_npu
import torch.distributed as dist

dist.init_process_group(backend="flagcx")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.npu.set_device(local_rank)

t = torch.randn(1000, dtype=torch.bfloat16, device="npu")
w = dist.all_reduce(t, async_op=True)
print(f"[sem] rank{rank} immediately is_completed={w.is_completed()}", flush=True)
torch.npu.synchronize()
print(f"[sem] rank{rank} after sync is_completed={w.is_completed()}", flush=True)
w.wait()
torch.npu.synchronize()
print(f"[sem] rank{rank} after wait is_completed={w.is_completed()}", flush=True)
dist.destroy_process_group()
print(f"[sem] rank{rank} done", flush=True)
