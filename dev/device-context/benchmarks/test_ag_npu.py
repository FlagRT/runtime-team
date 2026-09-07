"""
A 线 allgather 数据验证 —— torch_npu + FlagCX flagcx backend（910C）
验证：FlagCX 官方 HCCL 适配在 torch_npu 环境下的双卡 allgather 数据正确性
运行：
  cd /workspace/dev/device-context/benchmarks
  torchrun --nproc_per_node=2 --master_port=29511 test_ag_npu.py
"""
import os
import torch
import torch_npu  # noqa: F401  # 注册 npu 设备
import torch.distributed as dist

dist.init_process_group(backend="flagcx")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = dist.get_world_size()
torch.npu.set_device(local_rank)

t = torch.tensor([rank + 1], dtype=torch.int64, device="npu")
out = [torch.zeros(1, dtype=torch.int64, device="npu") for _ in range(world_size)]
dist.all_gather(out, t)
print(f"[agtest] rank{rank}: out={[x.item() for x in out]}", flush=True)

# 第二次调用（验证重复通信）
t2 = torch.tensor([rank + 10], dtype=torch.int64, device="npu")
out2 = [torch.zeros(1, dtype=torch.int64, device="npu") for _ in range(2)]
dist.all_gather(out2, t2)
print(f"[agtest] rank{rank}: out2={[x.item() for x in out2]}", flush=True)

dist.destroy_process_group()
