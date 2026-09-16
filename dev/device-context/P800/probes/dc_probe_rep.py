"""重复集合通信探针（参数化，用于定因）。

MODE:
  ar        重复 all_reduce(SUM)，每 10 次做一次设备同步  ← 基线
  ar_nosync 重复 all_reduce(SUM)，循环内不做设备同步
  ar_max    重复 all_reduce(MAX)（排除「值随次数增长/溢出」这一变量）
  barrier   只做 dist.barrier()
"""
import os, sys, time
import torch
import torch.distributed as dist
from datetime import timedelta
import flagcx  # noqa: F401  注册 cuda:flagcx

MODE = os.environ.get("MODE", "ar")
N = int(os.environ.get("REPS", "120"))
SIZE = int(os.environ.get("SIZE", "1024"))
SYNC_EVERY = int(os.environ.get("SYNC_EVERY", "10"))
lr = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
dev = f"cuda:{lr}"

dist.init_process_group("cpu:gloo,cuda:flagcx", timeout=timedelta(seconds=120))
rank = dist.get_rank()
print(f"[rank{rank}] PG ready | MODE={MODE} N={N} SIZE={SIZE} "
      f"KL3={os.environ.get('XPU_EVENT_KL3_ENABLE','<unset>')} "
      f"GCMASK={os.environ.get('BKCL_GC_SIGNAL_MASK','<unset>')} "
      f"cards={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

t = torch.ones(SIZE, SIZE, device=dev)
t0 = time.time()
for i in range(N):
    if MODE == "barrier":
        dist.barrier()
    elif MODE == "ar_max":
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    else:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if MODE == "ar_nosync":
            pass
        elif (i + 1) % SYNC_EVERY == 0:
            torch.cuda.synchronize()
    if (i + 1) % 10 == 0:
        print(f"[rank{rank}] rep={i+1} elapsed={time.time()-t0:.2f}s", flush=True)

if MODE != "ar_nosync":
    torch.cuda.synchronize()
print(f"[rank{rank}] ALLDONE mode={MODE} reps={N} elapsed={time.time()-t0:.2f}s", flush=True)
dist.destroy_process_group()
