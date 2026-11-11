import os, torch, torch_fl, torch.distributed as dist
dist.init_process_group(backend="flagos")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.flagos.set_device(local_rank)
t = torch.tensor([rank + 1], dtype=torch.int64, device="flagos")
out = [torch.zeros(1, dtype=torch.int64, device="flagos") for _ in range(dist.get_world_size())]
dist.all_gather(out, t)
print(f"[agtest] rank{rank}: out={[x.item() for x in out]}", flush=True)
# 再测一次（第二次调用）
t2 = torch.tensor([rank + 10], dtype=torch.int64, device="flagos")
out2 = [torch.zeros(1, dtype=torch.int64, device="flagos") for _ in range(2)]
dist.all_gather(out2, t2)
print(f"[agtest] rank{rank}: out2={[x.item() for x in out2]}", flush=True)
