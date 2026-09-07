"""4090-1 allgather 验证 —— FlagCX flagcx backend（NVIDIA adaptor）
预期：out=[1,2] / out2=[10,11]（两 rank 一致）
运行：torchrun --nproc_per_node=2 --master_port=29511 test_ag_cuda.py
"""
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="flagcx")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = dist.get_world_size()
torch.cuda.set_device(local_rank)

t = torch.tensor([rank + 1], dtype=torch.int64, device="cuda")
out = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(world_size)]
dist.all_gather(out, t)
print(f"[agtest] rank{rank}: out={[x.item() for x in out]}", flush=True)

t2 = torch.tensor([rank + 10], dtype=torch.int64, device="cuda")
out2 = [torch.zeros(1, dtype=torch.int64, device="cuda") for _ in range(2)]
dist.all_gather(out2, t2)
print(f"[agtest] rank{rank}: out2={[x.item() for x in out2]}", flush=True)

dist.destroy_process_group()
