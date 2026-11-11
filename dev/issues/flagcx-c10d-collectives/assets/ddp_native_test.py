#!/usr/bin/env python3
"""原生 DistributedDataParallel 双卡自判定测试（FlagCX c10d，2 ranks，自判定）

    torchrun --nproc_per_node=2 ddp_native_test.py [--steps N]

【为什么需要它】dev-10 的 issue `flagcx-c10d-collectives` 把「原生 DDP 不可用」列为
类别 ① 的**后果**：DDP 封装时会做 `_verify_params_across_processes`（走
`all_gather_into_tensor`），在 C++ 层命中「collective 返回时数据未写完」⇒
报 "rank0 有 338 个参数、其他 rank 0 个"。issue 的「修复验收」明确要求
**另需原生 DDP 双卡验证通过**（该脚本此前标注"待补"）。

【判据（三态，与 CASE-TEMPLATE §2 规则 3 一致）】
  0 `overall: PASS`  : DDP 封装成功 ∧ 多步训练不崩 ∧ **各 rank 参数逐位一致**（= 梯度确实被 allreduce）
  1 `overall: FAIL`  : 上述任一条不成立（缺陷存在或修复不完整）
  2 `overall: ERROR` : 无法下判定（环境/初始化失败等），不计入 PASS/FAIL
"""
import math
import os
import sys

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
from torch.nn.parallel import DistributedDataParallel as DDP

STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 10


def main() -> int:
    try:
        dist.init_process_group(backend="flagcx")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: init_process_group(flagcx) 失败: {type(e).__name__}: {e}", flush=True)
        print("overall: ERROR", flush=True)
        return 2

    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(lr)
    dev = f"npu:{lr}"

    # ── 1) DDP 封装本身（类别 ① 的原始触发点就在这里）──
    try:
        torch.manual_seed(1234)                       # 各 rank 同初值 ⇒ 参数应始终保持一致
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)
        ).to(dev)
        ddp = DDP(model)
        wrapped = True
    except Exception as e:  # noqa: BLE001
        print(f"[rank{rank}] DDP 封装失败: {type(e).__name__}: {str(e)[:300]}", flush=True)
        wrapped = False

    # ── 2) 多步训练 ──
    losses, crashed = [], None
    if wrapped:
        try:
            opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
            x = torch.randn(16, 8, device=dev)
            y = torch.randn(16, 1, device=dev)
            for _ in range(STEPS):
                opt.zero_grad()
                loss = ((ddp(x) - y) ** 2).mean()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        except Exception as e:  # noqa: BLE001
            crashed = f"{type(e).__name__}: {str(e)[:200]}"

    # ── 3) 各 rank 参数是否逐位一致（allreduce 真生效的判据）──
    same = False
    detail = ""
    if wrapped and crashed is None:
        try:
            torch.npu.synchronize()
            flat = torch.cat([p.detach().flatten() for p in ddp.parameters()])
            mx = flat.clone()
            mn = flat.clone()
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            dist.all_reduce(mn, op=dist.ReduceOp.MIN)
            same = bool(torch.equal(mx, mn))
            detail = f"param_max_min_equal={same} sum={float(flat.sum()):.6f}"
        except Exception as e:  # noqa: BLE001
            crashed = f"一致性检查失败: {type(e).__name__}: {str(e)[:200]}"

    finite = bool(losses) and all(math.isfinite(v) for v in losses)

    # ── 汇总（与 flagcx_sync_test.py 同口径：打印后 os._exit，避开已知退出期 SIGABRT）──
    if rank == 0:
        print(f"\n=== DDP VERDICT (steps={STEPS}, world_size={ws}) ===", flush=True)
        print(f"  ddp_wrapped        {'PASS' if wrapped else 'FAIL'}", flush=True)
        print(f"  train_steps        {'PASS' if (crashed is None and len(losses) == STEPS) else 'FAIL'}", flush=True)
        print(f"  loss_all_finite    {'PASS' if finite else 'FAIL'}  first={losses[0]:.6f} last={losses[-1]:.6f}" if losses else
              f"  loss_all_finite    FAIL", flush=True)
        print(f"  ranks_param_equal  {'PASS' if same else 'FAIL'}  {detail}", flush=True)
        if crashed:
            print(f"  crashed_at         {crashed}", flush=True)
        ok = bool(wrapped and crashed is None and len(losses) == STEPS and finite and same)
        print(f"overall: {'PASS' if ok else 'FAIL'}", flush=True)
    else:
        ok = True  # 非 0 rank 由 rank0 判定（与 flagcx_sync_test.py 同：只 rank0 打印）
        ok = bool(wrapped and crashed is None and len(losses) == STEPS and finite and same)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    sys.exit(main())
