"""验证探针：确认「通过 / 挂死」是真结论，而不是观测假阴性。

关键设计：无论循环里是否做设备同步，**结尾都做一次同步并校验张量真值** ——
各 rank 的初值全 1，经 SUM all_reduce 后每轮翻 world 倍，
故 N 轮后应恰为 2^N（float32 可精确表示 2 的幂）。
真值正确 ⇒ 集合通信确实执行且结果正确；真值不对 ⇒ 「通过」是假的。
"""
import math
import os
import time

import torch
import torch.distributed as dist
from datetime import timedelta
import flagcx  # noqa: F401  注册 cuda:flagcx

MODE = os.environ.get("MODE", "sync10")
N = int(os.environ.get("REPS", "120"))
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dev = f"cuda:{lr}"

dist.init_process_group("cpu:gloo,cuda:flagcx", timeout=timedelta(seconds=120))
rank = dist.get_rank()
world = dist.get_world_size()
print(f"[rank{rank}] MODE={MODE} N={N} world={world} "
      f"KL3={os.environ.get('XPU_EVENT_KL3_ENABLE','<unset>')}", flush=True)

t = torch.ones(1024, 1024, device=dev)
t0 = time.time()
for i in range(N):
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if MODE == "sync10" and (i + 1) % 10 == 0:
        torch.cuda.synchronize()
    if (i + 1) % 20 == 0:
        print(f"[rank{rank}] rep={i+1} elapsed={time.time()-t0:.2f}s", flush=True)

torch.cuda.synchronize()
v = float(t[0, 0].item())
exp = float(world) ** N
rel = abs(v - exp) / exp if exp else float("inf")
print(f"[rank{rank}] VERIFY value={v:.6e} expected=2^{N*math.log2(world):.0f}={exp:.6e} "
      f"rel_err={rel:.3e} exact={rel < 1e-6}", flush=True)
print(f"[rank{rank}] ALLDONE mode={MODE} reps={N} elapsed={time.time()-t0:.2f}s", flush=True)
dist.destroy_process_group()
