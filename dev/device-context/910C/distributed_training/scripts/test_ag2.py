import os, torch, torch_fl, torch.distributed as dist
dist.init_process_group(backend="flagos")
rank = dist.get_rank()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.flagos.set_device(local_rank)
# 用 flagos tensor 作为 send/recv（先写数据）
t = torch.tensor([rank + 1], dtype=torch.int64, device="flagos")
out = [torch.zeros(1, dtype=torch.int64, device="flagos") for _ in range(2)]
print(f"[t2] rank{rank}: send={t.item()} ptr={hex(t.data_ptr())}", flush=True)
dist.all_gather(out, t)
print(f"[t2] rank{rank}: out={[x.item() for x in out]}", flush=True)
# 显式同步
torch.flagos.synchronize()
print(f"[t2] rank{rank}: after sync out={[x.item() for x in out]}", flush=True)
# 直接读 outputFlattened 对应内存？不行。用 hccl_py_smoke 类似方式：检查 recv buffer
