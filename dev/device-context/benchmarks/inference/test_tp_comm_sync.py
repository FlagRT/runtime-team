#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_tp_comm_sync.py — TP 通信语义验证（P2 核心：职责 D5 Stream / D9 同步语义）
═══════════════════════════════════════════════════════════════════════════════

【背景】B 线（torch_fl）TP 推理踩过的两个根因，A 线（torch_npu）需重验：
  B2: flagcx 后端 all_reduce 异步返回无同步 → 下游消费未就绪数据 → NaN
  A7: TP 通信流绑定——collective 是否落在当前流（与下游计算有 happens-before）

【判据】（关键设计：不能靠 .cpu() 检测未就绪，因为拷贝本身会隐式同步）
  用**设备侧后续计算**暴露依赖：all_reduce 后立即在同一流做 t*2，
  若流绑定正确 → 结果 = 归约后值 ×2；若 allreduce 不在当前流 → 读到未归约值。

【场景】
  A: all_reduce → 立即设备侧 t*2 → sync → 读（判 A7 流绑定）
  B: all_reduce → synchronize → t*2 → sync → 读（对照）
  C: async_op=True + work.wait() → t*2 → sync → 读（判 work 完成语义）
  D: 连续 20 轮 all_reduce（模拟多轮 decode）→ 检查 NaN/数值漂移（判 B2）

【用法】容器内 A 线环境（torch_npu + FlagCX），双卡：
  torchrun --nproc_per_node=2 --master_port=29531 test_tp_comm_sync.py
【输出】stdout + tp_comm_sync_result.json（每 rank 一份，rank0 为准）
"""

import json
import os

import torch
import torch_npu
import torch.distributed as dist

N = 1000


def _val(t):
    """数值纪律：.cpu() 后读取（坑 B4：flag_gems sum 算子不可靠）。"""
    return t.cpu().clone()


def main():
    dist.init_process_group(backend="flagcx")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.npu.set_device(local_rank)
    world = dist.get_world_size()
    expect_base = sum(r + 1 for r in range(world))  # rank0=1, rank1=2 → 3

    res = {"rank": rank, "world": world, "expect_base": expect_base, "checks": {}}

    def fresh():
        # 每 rank 初始值 = rank+1；all_reduce 后应为 expect_base
        return torch.ones(N, dtype=torch.float32, device="npu") * (rank + 1)

    # ── A: all_reduce 后立即设备侧消费（流绑定判据）──
    t = fresh()
    dist.all_reduce(t)
    out_a = (t * 2)                      # 当前流上的后续计算（依赖归约结果）
    torch.npu.synchronize()
    va = _val(out_a)[0].item()
    ok_a = abs(va - expect_base * 2) < 1e-3
    res["checks"]["a_stream_binding"] = {
        "ok": ok_a, "got": va, "expect": expect_base * 2,
        "detail": f"all_reduce 后立即设备侧消费: got={va} expect={expect_base*2}"}
    print(f"[rank{rank}][A] 流绑定: got={va} expect={expect_base*2} ok={ok_a}", flush=True)

    # ── B: 对照组（显式 synchronize 后再消费）──
    t = fresh()
    dist.all_reduce(t)
    torch.npu.synchronize()
    out_b = t * 2
    torch.npu.synchronize()
    vb = _val(out_b)[0].item()
    ok_b = abs(vb - expect_base * 2) < 1e-3
    res["checks"]["b_with_sync"] = {
        "ok": ok_b, "got": vb, "expect": expect_base * 2,
        "detail": f"同步后消费: got={vb} expect={expect_base*2}"}
    print(f"[rank{rank}][B] 同步对照: got={vb} ok={ok_b}", flush=True)

    # ── C: async_op + work.wait() ──
    t = fresh()
    w = dist.all_reduce(t, async_op=True)
    w.wait()
    out_c = t * 2
    torch.npu.synchronize()
    vc = _val(out_c)[0].item()
    ok_c = abs(vc - expect_base * 2) < 1e-3
    res["checks"]["c_async_work_wait"] = {
        "ok": ok_c, "got": vc, "expect": expect_base * 2,
        "detail": f"async_op+work.wait() 后消费: got={vc} expect={expect_base*2}"}
    print(f"[rank{rank}][C] async+wait: got={vc} ok={ok_c}", flush=True)

    # ── D: 连续 20 轮（模拟多轮 decode，每轮新张量），检测 NaN / 归约正确性（B2 判据）
    #    注：若用同一张量连续 all_reduce，值会按 3×2^(n-1) 增长（非缺陷，是归约叠加）
    nan_round, drift = -1, 0.0
    last = None
    for i in range(20):
        t = fresh()                      # 每轮新张量：模拟每轮 decode 的独立归约
        dist.all_reduce(t)
        last = t
    torch.npu.synchronize()
    vd = _val(last)
    has_nan = bool(torch.isnan(vd).any().item())
    first = vd[0].item()
    drift = abs(first - expect_base)
    ok_d = (not has_nan) and drift < 1e-2
    res["checks"]["d_multiround"] = {
        "ok": ok_d, "has_nan": has_nan, "first_val": first,
        "expect": expect_base, "drift": drift,
        "detail": f"20 轮 all_reduce: NaN={has_nan} 首值={first} (expect={expect_base})"}
    print(f"[rank{rank}][D] 20 轮: NaN={has_nan} 首值={first} expect={expect_base} ok={ok_d}",
          flush=True)

    passed = sum(1 for v in res["checks"].values() if v["ok"])
    res["summary"] = {"passed": passed, "total": len(res["checks"])}
    res["verdict"] = "TP_COMM_PASS" if passed == len(res["checks"]) else "TP_COMM_PARTIAL"
    print(f"[rank{rank}] {res['verdict']}: {passed}/{len(res['checks'])}", flush=True)

    if rank == 0:
        with open("tp_comm_sync_result.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
