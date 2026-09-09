#!/usr/bin/env python3
"""flagcx 2 进程 allreduce 冒烟(验证 c10_npu shim 的 stream/event 路径)。

用法(容器内):
  ASCEND_RT_VISIBLE_DEVICES=0,1 python flagcx_smoke.py 0 2 &
  ASCEND_RT_VISIBLE_DEVICES=0,1 python flagcx_smoke.py 1 2 &
期望: 两个 rank 都打印 allreduce OK, 数值 = rank 值之和。
"""
import os
import sys

import torch
import torch.distributed as dist
import flagcx  # noqa: F401  — 触发 _C 加载与 torch.distributed 后端注册

rank = int(sys.argv[1])
world = int(sys.argv[2])
dev = sys.argv[3] if len(sys.argv) > 3 else str(rank)  # ASCEND_RT_VISIBLE_DEVICES 单卡(TP 语义)

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", dev)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29555")
os.environ.setdefault("RANK", str(rank))
os.environ.setdefault("WORLD_SIZE", str(world))

dist.init_process_group(backend="flagcx", rank=rank, world_size=world)
print(f"[rank{rank}] init OK, backend={dist.get_backend()}")

t = torch.full((4,), rank + 1, dtype=torch.int32, device="flagos:0")
print(f"[rank{rank}] before:", t.cpu().tolist())
dist.all_reduce(t)
torch.npu.synchronize()  # flagcx 后端异步返回,需显式同步(阶段4 根因2 修法)
print(f"[rank{rank}] after :", t.cpu().tolist())
expected = world * (world + 1) // 2
ok = t.cpu().tolist() == [expected] * 4
print(f"[rank{rank}] allreduce {'OK' if ok else 'FAIL'} (expected {expected})")
dist.destroy_process_group()
print(f"[rank{rank}] done")
