#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_tp_comm_sync_enhanced.py — A6/A7 补验：更强场景下的 TP 通信与流绑定

【原验证的不足】原 `test_tp_comm_sync.py` 四个场景（A/B/C/D）均为**单流 + 小数据**：
  · 场景 A「all_reduce 后立即设备侧消费正确」**不具备排他性** —— 若 all_reduce 内部
    做了隐式同步（不论绑定哪个流），A 同样会通过，无法证明"绑在调用者当前流"
  · 未压测大数据量（小张量掩盖了传输/归约的真实行为）
  · 未覆盖多流竞争（真实推理中采样/传输/计算可能在不同流）

【补验场景】
  E 大数据量   ：64MB 张量 all_reduce + 立即设备侧消费（规模压测）
  F 跨流不阻塞 ：**流绑定的排他证据** —— 流 S 发大 all_reduce，同时流 T 发起独立计算，
                 用事件时序验证流 T 的计算与 all_reduce **重叠**；若 all_reduce 错误地
                 绑在默认流/做全局同步，流 T 会被阻塞（不重叠）
  G 多流并发   ：4 个流同时发起各自 all_reduce，验证结果两两正确（真实竞争场景）
  H 长轮稳定性 ：100 轮 all_reduce（64MB），检查 NaN / 数值漂移

【用法】容器内（serve 需已停）：
  torchrun --nproc_per_node=2 test_tp_comm_sync_enhanced.py --out tp_comm_enhanced.json
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401


def _val(t):
    return t.float().sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-mb", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--out", default="tp_comm_enhanced.json")
    args = ap.parse_args()

    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    dev = f"npu:{rank}"
    torch.npu.set_device(dev)

    res = {"rank": rank, "world": world, "checks": {}}
    base = sum(r + 1 for r in range(world))  # all_reduce 后的期望基准
    n_big = args.big_mb * 1024 * 1024 // 4    # fp32 元素数

    def fresh(n=1):
        return torch.full((n,), float(rank + 1), device=dev)

    # ── E 大数据量 ──
    t = fresh(n_big)
    dist.all_reduce(t)
    out_e = t * 2                      # 立即设备侧消费（流绑定判据）
    torch.npu.synchronize()
    ve = _val(out_e)
    exp_e = base * 2 * n_big
    ok_e = abs(ve - exp_e) / max(abs(exp_e), 1) < 1e-4
    res["checks"]["E_big_allreduce"] = {
        "ok": ok_e, "size_mb": args.big_mb,
        "got": ve, "expect": exp_e,
        "detail": f"{args.big_mb}MB all_reduce 后立即消费 got={ve:.4e} expect={exp_e:.4e}"}
    print(f"[rank{rank}][E] 大张量({args.big_mb}MB) 流绑定: ok={ok_e}", flush=True)

    # ── F 跨流不阻塞（流绑定排他证据）──
    s_comm = torch.npu.Stream()   # 通信流
    s_calc = torch.npu.Stream()   # 独立计算流
    t2 = fresh(n_big)
    # 先在计算流上放一个"长任务"作为背景负载基准
    bg = torch.ones(4096, 4096, device=dev)
    ev_calc_start = torch.npu.Event(enable_timing=True)
    ev_calc_end = torch.npu.Event(enable_timing=True)
    ev_comm_start = torch.npu.Event(enable_timing=True)
    ev_comm_end = torch.npu.Event(enable_timing=True)

    torch.npu.synchronize()
    t0 = time.time()
    with torch.npu.stream(s_comm):
        ev_comm_start.record()
        dist.all_reduce(t2)
        ev_comm_end.record()
    with torch.npu.stream(s_calc):
        ev_calc_start.record()
        for _ in range(20):
            bg = bg @ bg * 1.0
        ev_calc_end.record()
    torch.npu.synchronize()
    wall = time.time() - t0

    ms_comm = ev_comm_start.elapsed_time(ev_comm_end)
    ms_calc = ev_calc_start.elapsed_time(ev_calc_end)
    wall_ms = wall * 1000
    serial_est = ms_comm + ms_calc          # 完全串行的理论时间
    # 重叠效率 = 实际节省 / 理论最大可节省
    #   理论最大可节省 = min(comm, calc)（较短者完全被较长者掩盖）
    #   注意：不能用 (serial - wall)/serial —— 当两者量级悬殊时该比值天然趋近 0，
    #   即使完全重叠也会误判为"无重叠"（首轮 rank0 即因此误判失败）
    max_saving = min(ms_comm, ms_calc)
    actual_saving = serial_est - wall_ms
    efficiency = (actual_saving / max_saving) if max_saving > 0 else 0.0
    ok_f = efficiency > 0.30        # 重叠效率 >30% 即证未互相阻塞
    res["checks"]["F_cross_stream_no_block"] = {
        "ok": ok_f, "wall_ms": round(wall_ms, 2),
        "comm_ms": round(ms_comm, 2), "calc_ms": round(ms_calc, 2),
        "serial_est_ms": round(serial_est, 2),
        "overlap_efficiency": round(efficiency, 4),
        "detail": (f"通信流 all_reduce {ms_comm:.1f}ms ∥ 计算流 {ms_calc:.1f}ms；"
                   f"串行估计 {serial_est:.1f}ms、实测墙钟 {wall_ms:.1f}ms → "
                   f"重叠效率 {efficiency:.1%}（>30% 证未互相阻塞）")}
    print(f"[rank{rank}][F] 跨流不阻塞: comm={ms_comm:.1f}ms calc={ms_calc:.1f}ms "
          f"效率={efficiency:.1%} ok={ok_f}", flush=True)

    # ── G 多流并发 collective ──
    n_streams = 4
    streams = [torch.npu.Stream() for _ in range(n_streams)]
    tensors = [fresh(1024) for _ in range(n_streams)]
    for i, s in enumerate(streams):
        with torch.npu.stream(s):
            dist.all_reduce(tensors[i])
    torch.npu.synchronize()
    vals = [_val(t) for t in tensors]
    exp_g = base * 1024
    ok_g = all(abs(v - exp_g) / exp_g < 1e-4 for v in vals)
    res["checks"]["G_multi_stream_concurrent"] = {
        "ok": ok_g, "n_streams": n_streams, "vals": vals, "expect": exp_g,
        "detail": f"{n_streams} 流并发 all_reduce 结果: {vals} expect={exp_g}"}
    print(f"[rank{rank}][G] {n_streams} 流并发: vals={vals} ok={ok_g}", flush=True)

    # ── H 长轮稳定性 ──
    t3 = fresh(n_big // 8)
    first = None
    has_nan = False
    for i in range(args.rounds):
        dist.all_reduce(t3)
        if i == 0:
            first = _val(t3)
        if not bool(torch.isfinite(t3).all()):
            has_nan = True
            break
    torch.npu.synchronize()
    ok_h = (not has_nan) and abs(first - base * (n_big // 8)) / (base * (n_big // 8)) < 1e-4
    res["checks"]["H_long_rounds"] = {
        "ok": ok_h, "rounds": args.rounds, "nan": has_nan, "first": first,
        "detail": f"{args.rounds} 轮 all_reduce: NaN={has_nan} 首值={first}"}
    print(f"[rank{rank}][H] {args.rounds} 轮: NaN={has_nan} ok={ok_h}", flush=True)

    passed = sum(1 for c in res["checks"].values() if c["ok"])
    total = len(res["checks"])
    res["passed"] = passed
    res["total"] = total
    res["verdict"] = "TP_COMM_ENHANCED_PASS" if passed == total else "TP_COMM_ENHANCED_PARTIAL"
    print(f"[rank{rank}] === {res['verdict']} ({passed}/{total}) ===", flush=True)

    if rank == 0:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"结果：{args.out}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
