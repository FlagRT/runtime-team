"""最小复现：torchrun 是否能在本容器跑起 2 进程 + nccl 设备通信。

每步都 flush 打印，便于定位卡在哪一步。
"""
import os
import sys

print(f"[boot] rank={os.environ.get('RANK')} local_rank={os.environ.get('LOCAL_RANK')} "
      f"world={os.environ.get('WORLD_SIZE')} master={os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}",
      flush=True)

import torch  # noqa: E402

print(f"[import] torch={torch.__version__} cuda_avail={torch.cuda.is_available()} "
      f"count={torch.cuda.device_count()}", flush=True)

import torch.distributed as dist  # noqa: E402

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
print(f"[set_device] rank={rank} local={local_rank} cur={torch.cuda.current_device()} "
      f"name={torch.cuda.get_device_name(local_rank)}", flush=True)

print("[init_pg] before", flush=True)
dist.init_process_group("nccl")
print(f"[init_pg] after backend={dist.get_backend()} ws={dist.get_world_size()}", flush=True)

t = torch.full((8,), float(rank + 1), device="cuda")
dist.all_reduce(t, op=dist.ReduceOp.SUM)
print(f"[allreduce] rank={rank} got={t[0].item()} expect={sum(range(1, 3))}", flush=True)

dist.barrier()
dist.destroy_process_group()
print(f"[done] rank={rank}", flush=True)
sys.exit(0)
