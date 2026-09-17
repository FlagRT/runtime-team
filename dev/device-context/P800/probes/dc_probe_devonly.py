"""对照组：单进程、纯设备计算（不做任何通信），检验设备本身是否稳定。"""
import os, time
import torch
dev = "cuda:0"
x = torch.ones(1024, 1024, device=dev)
t0 = time.time()
N = int(os.environ.get("REPS", "2000"))
for i in range(N):
    x.add_(1e-7)                       # 原地加，避免显存增长
    if (i + 1) % 200 == 0:
        torch.cuda.synchronize()
        if float(x[0, 0]) > 1e6:
            x.fill_(1.0)
        print(f"devonly rep={i+1} elapsed={time.time()-t0:.2f}s val={float(x[0,0]):.4f}", flush=True)
torch.cuda.synchronize()
print(f"ALLDONE devonly reps={N} elapsed={time.time()-t0:.2f}s", flush=True)
