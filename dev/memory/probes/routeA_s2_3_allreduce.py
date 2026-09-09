#!/usr/bin/env python
"""S2.3 FlagCX 单机多卡 allreduce 验证（Route A / P800）

用法（容器内）:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=1,2 FLAGCX_ADAPTOR=klx \
      torchrun --nproc-per-node=2 /workspace/dev/memory/probes/routeA_s2_3_allreduce.py

覆盖: flagcx backend 初始化 / allreduce 正确性 / 多尺寸带宽
"""
import os
import time

import torch
import torch.distributed as dist

import flagcx  # noqa: F401  # 注册 cuda:flagcx backend

RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(LOCAL_RANK)
dev = f"cuda:{LOCAL_RANK}"

# backend 组合: cpu 上用 gloo, 设备上用 flagcx
dist.init_process_group("cpu:gloo,cuda:flagcx", rank=RANK, world_size=WORLD_SIZE)

if RANK == 0:
    print("=" * 70)
    print("S2.3 FlagCX allreduce (FLAGCX_ADAPTOR=%s)" % os.environ.get("FLAGCX_ADAPTOR", "(未设置,依赖编译期适配器)"))
    print(f"torch={torch.__version__} flagcx={flagcx.__version__ if hasattr(flagcx, '__version__') else '?'} world={WORLD_SIZE}")
    print("=" * 70)

# --- 正确性 ---
x = torch.arange(8, dtype=torch.float32, device=dev) + RANK
dist.all_reduce(x, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
# rank r 贡献 arange(8)+r, 求和 = W*arange(8) + sum(range(W))
expected = WORLD_SIZE * torch.arange(8, dtype=torch.float32, device=dev) + sum(range(WORLD_SIZE))
ok = torch.equal(x, expected)
print(f"[rank {RANK}] allreduce 正确性: {'PASS' if ok else 'FAIL'}  x={x.tolist()}")
dist.barrier()

# --- 带宽 ---
sizes = [128 * 1024 * 1024, 512 * 1024 * 1024, 1024 * 1024 * 1024]  # 128MB/512MB/1GB
for nbytes in sizes:
    n = nbytes // 4
    # 每 rank 填充不同常数, allreduce SUM 后应全为 sum(range(W)+1)
    fill = float(RANK + 1)
    t = torch.full((n,), fill, dtype=torch.float32, device=dev)
    # warmup (原地累加会让 t 每轮翻倍)
    for _ in range(3):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()
    iters = 10
    # 计时前重置数据, 之后 t = expect * 2**(3+iters)
    t = torch.full((n,), fill, dtype=torch.float32, device=dev)
    t0 = time.time()
    for _ in range(iters):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()  # 阻塞到通信真正完成
    torch.cuda.synchronize()
    dist.barrier()
    dt = (time.time() - t0) / iters
    gbs = nbytes / dt / 1e9
    # 数据校验: 通信真实发生。注意: 对全新缓冲的第一次 allreduce 产生基数
    # sum(range(W)+1)=3 (1+2), 之后每次翻倍 -> k 次后 = 3 * 2^(k-1)
    expect = sum(range(WORLD_SIZE + 1)) * (2 ** (iters - 1))  # = 3 * 2^9 = 1536
    ok = bool((t[:4] == expect).all().item()) and abs(t.sum().item() / n - expect) < expect * 1e-3
    print(f"[rank {RANK}] allreduce {nbytes//(1024*1024)}MB x{iters}: {dt*1e6:.1f} us/iter -> {gbs:.2f} GB/s (2x 口径 {2*gbs:.2f}) 数据校验:{'PASS' if ok else 'FAIL'}")

dist.destroy_process_group()
if RANK == 0:
    print("S2.3 FlagCX allreduce 完成")
