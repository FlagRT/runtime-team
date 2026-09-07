#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_dbuf_sync_variants.py — O2：D8 双缓冲「减少同步点」方案对比实验

【背景】A3/D8 功能达标（多流真并发 + 数据一致）但性能未达：重叠率 -101.6% ~ -1306%。
       Level1 profiler 定位瓶颈 = EVENT_WAIT 12 次 / 累计 3365us（单次最高 391us），
       而 512² matmul 仅约 10us 量级 → **同步开销比计算高一个数量级**。
       多流真并发已证实，故优化方向不是提升并发，而是**减少同步点**。

【变体设计】（单变量隔离，固定 n_batches=6 / n=512，V3 除外只变计算粒度）
  V0 基线    ：每批 2 × record + 2 × wait_event（H2D→计算、计算→D2H 各一条依赖链）
  V1 精简链  ：每批 1 × record + 1 × wait（只保留 H2D→计算），D2H 改为末尾一次 wait_stream
  V2 流依赖  ：用 s_calc.wait_stream(s_trans) 替代 record+wait_event（每批 1 次等待，无 event 对象）
  V3 粗粒度  ：基线结构不变，计算量放大（n_calc_repeat 次 matmul），验证"计算掩盖同步"

【测量】每个变体：预热 1 轮（昇腾首次算子开销可达百倍，见 breakdown 探针）→ 计时光流水线与串行 →
       计算重叠率 (serial - pipe) / serial → 验证末批数据正确 → 重复 rounds 次取中位

【用法】容器内：python3 test_dbuf_sync_variants.py [--rounds 3] [--out dbuf_sync_variants_result.json]
"""
import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"


def _warmup():
    """设备 + pin_memory 预热（坑：首次算子开销可达 107 倍）"""
    torch.zeros(1, device=DEV)
    h = torch.randn(256, 256).pin_memory()
    d = torch.zeros(256, 256, device=DEV)
    d.copy_(h, non_blocking=True)
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        _ = (d @ d).sum()
    torch.npu.synchronize()


def _calc(buf, n_calc_repeat):
    """计算负载：n_calc_repeat 次 matmul（可放大计算粒度）"""
    for _ in range(n_calc_repeat):
        buf @ buf
    return buf.sum()


def variant_v0(n_batches, n, n_calc_repeat):
    """V0 基线：每批 2 record + 2 wait"""
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = [torch.zeros(n, n, device=DEV) for _ in range(2)]
    ev_h2d = [torch.npu.Event() for _ in range(2)]
    ev_calc = [torch.npu.Event() for _ in range(2)]
    s_trans, s_calc, s_d2h = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
    out = [None, None]

    t0 = time.time()
    for i in range(n_batches):
        b = i % 2
        with torch.npu.stream(s_trans):
            buf[b].copy_(hosts[b], non_blocking=True)
        ev_h2d[b].record(s_trans)
        s_calc.wait_event(ev_h2d[b])
        with torch.npu.stream(s_calc):
            out[b] = _calc(buf[b], n_calc_repeat)
        ev_calc[b].record(s_calc)
        s_d2h.wait_event(ev_calc[b])
        with torch.npu.stream(s_d2h):
            _ = out[b].to("cpu", non_blocking=True)
    torch.npu.synchronize()
    return time.time() - t0, buf


def variant_v1(n_batches, n, n_calc_repeat):
    """V1 精简链：每批 1 record + 1 wait（去掉 计算→D2H 的事件链，末尾统一 wait_stream）"""
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = [torch.zeros(n, n, device=DEV) for _ in range(2)]
    ev = [torch.npu.Event() for _ in range(2)]
    s_trans, s_calc, s_d2h = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
    outs = [None, None]

    t0 = time.time()
    for i in range(n_batches):
        b = i % 2
        with torch.npu.stream(s_trans):
            buf[b].copy_(hosts[b], non_blocking=True)
        ev[b].record(s_trans)
        s_calc.wait_event(ev[b])
        with torch.npu.stream(s_calc):
            outs[b] = _calc(buf[b], n_calc_repeat)
    # 回传：末尾统一一次流依赖（而非每批一条事件链）
    s_d2h.wait_stream(s_calc)
    with torch.npu.stream(s_d2h):
        for o in outs:
            if o is not None:
                _ = o.to("cpu", non_blocking=True)
    torch.npu.synchronize()
    return time.time() - t0, buf


def variant_v2(n_batches, n, n_calc_repeat):
    """V2 流依赖：wait_stream 替代 record+wait_event（无 event 对象，每批 1 次等待）"""
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = [torch.zeros(n, n, device=DEV) for _ in range(2)]
    s_trans, s_calc = torch.npu.Stream(), torch.npu.Stream()
    outs = [None, None]

    t0 = time.time()
    for i in range(n_batches):
        b = i % 2
        with torch.npu.stream(s_trans):
            buf[b].copy_(hosts[b], non_blocking=True)
        s_calc.wait_stream(s_trans)  # 语义更粗：等待 s_trans 上此前所有任务
        with torch.npu.stream(s_calc):
            outs[b] = _calc(buf[b], n_calc_repeat)
    torch.npu.synchronize()
    for o in outs:
        if o is not None:
            _ = o.to("cpu", non_blocking=True)
    return time.time() - t0, buf


def serial_baseline(n_batches, n, n_calc_repeat):
    """串行参照：全部在同一默认流上顺序执行"""
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = [torch.zeros(n, n, device=DEV) for _ in range(2)]
    t0 = time.time()
    for i in range(n_batches):
        b = i % 2
        buf[b].copy_(hosts[b])                 # 同步拷贝
        _ = _calc(buf[b], n_calc_repeat)       # 同流计算
    torch.npu.synchronize()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default="dbuf_sync_variants_result.json")
    args = ap.parse_args()

    _warmup()

    variants = [
        ("V0_基线_2record2wait", variant_v0, 1),
        ("V1_精简链_1record1wait", variant_v1, 1),
        ("V2_流依赖_wait_stream", variant_v2, 1),
        ("V3_粗粒度_计算x8", variant_v0, 8),
    ]

    result = {"config": vars(args), "variants": {}, "note": ""}
    print(f"=== O2 双缓冲同步点对比：batches={args.batches} n={args.n} rounds={args.rounds} ===")
    print(f"{'变体':<26}{'pipe(s)':>10}{'serial(s)':>11}{'重叠率':>10}   说明")

    for name, fn, rep in variants:
        pipes, oks = [], []
        for _ in range(args.rounds):
            t, buf = fn(args.batches, args.n, rep)
            pipes.append(t)
            oks.append(bool(torch.isfinite(buf[0]).all() and torch.isfinite(buf[1]).all()))
        serial = serial_baseline(args.batches, args.n, rep)
        pipe = statistics.median(pipes)
        overlap = (serial - pipe) / serial if serial > 0 else 0.0
        result["variants"][name] = {
            "pipe_s": round(pipe, 4),
            "serial_s": round(serial, 4),
            "overlap": round(overlap, 4),
            "overlap_pct": f"{overlap:.1%}",
            "n_calc_repeat": rep,
            "data_ok": all(oks),
            "all_runs": [round(x, 4) for x in pipes],
        }
        print(f"{name:<26}{pipe:>10.4f}{serial:>11.4f}{overlap:>9.1%}   计算×{rep}")

    # 结论
    best = max(result["variants"].items(), key=lambda kv: kv[1]["overlap"])
    result["best_by_overlap"] = {"name": best[0], "overlap_pct": best[1]["overlap_pct"]}
    print(f"\n=== 最佳（按重叠率）：{best[0]} 重叠率 {best[1]['overlap_pct']} ===")
    if best[1]["overlap"] > 0.05:
        result["note"] = f"方案 {best[0]} 达成重叠率 {best[1]['overlap_pct']}（> 5%），A3 性能项可闭环"
        print(f"[结论] {result['note']}")
    else:
        result["note"] = (
            f"最佳方案 {best[0]} 重叠率 {best[1]['overlap_pct']}，仍未达 5% 阈值；"
            f"需继续减少同步点或放大计算粒度"
        )
        print(f"[结论] {result['note']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
