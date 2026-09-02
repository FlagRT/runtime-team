#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_dbuf_variants_rigorous.py — D8 求解实验：找到能让重叠率为正的同步方案

【前情】前一轮四变体对比用了不严谨测法（跨规模复用小尺寸预热），结论作废。
      严谨分段计时后已知：n=512 时 pipe=1.050ms，而「拷贝+计算」仅 0.337ms，
      多出的 0.713ms 全是同步开销；要让重叠率为正需把同步开销砍掉约 84%。

【本实验求解什么】同步开销到底花在哪、能否压下来
  · 分解：1 次 record/wait 的**单次成本**（用不同同步次数的变体反推）
  · 变体：
      V0 基线   每批 2 record+2 wait → 12 次同步操作（6 批）
      V1 精简链 每批 1 record+1 wait → 6 次
      V4 批量   全部拷贝→1 次同步→全部计算→1 次同步→全部回传 → 2 次（牺牲批间流水换同步次数）
      V5 无同步 全部在同一流顺序执行（理论下界参照，无重叠）
  · 每个变体同时给出：pipe、与「拷贝+计算」的差值（=同步开销）、两种口径重叠率

【判定】
  若某变体 pipe < (copy_sync + calc) → 重叠率为正，求解成功
  否则报告"同步开销下限"，并给出达成正值所需的计算粒度阈值（用扫描定位转折点）

【用法】容器内：python3 test_dbuf_variants_rigorous.py [--sizes 512,1024,2048,4096] [--rounds 7]
"""
import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"
BATCHES = 6


def _warmup_basic():
    torch.zeros(1, device=DEV)
    h = torch.randn(256, 256).pin_memory()
    d = torch.zeros(256, 256, device=DEV)
    d.copy_(h, non_blocking=True)
    torch.npu.synchronize()


def _calc(b, rep=1):
    for _ in range(rep):
        b @ b
    return b.sum()


# ---------------- 变体 ----------------
def v0(hosts, bufs, ev_h2d, ev_calc, s_t, s_c, s_d):
    for i in range(BATCHES):
        b = i % 2
        with torch.npu.stream(s_t):
            bufs[b].copy_(hosts[b], non_blocking=True)
        ev_h2d[b].record(s_t)
        s_c.wait_event(ev_h2d[b])
        with torch.npu.stream(s_c):
            r = _calc(bufs[b])
        ev_calc[b].record(s_c)
        s_d.wait_event(ev_calc[b])
        with torch.npu.stream(s_d):
            _ = r.to("cpu", non_blocking=True)


def v1(hosts, bufs, ev, s_t, s_c, s_d):
    outs = [None] * BATCHES
    for i in range(BATCHES):
        b = i % 2
        with torch.npu.stream(s_t):
            bufs[b].copy_(hosts[b], non_blocking=True)
        ev[b].record(s_t)
        s_c.wait_event(ev[b])
        with torch.npu.stream(s_c):
            outs[i] = _calc(bufs[b])
    s_d.wait_stream(s_c)
    with torch.npu.stream(s_d):
        for o in outs:
            _ = o.to("cpu", non_blocking=True)


def v4(hosts, bufs, ev, s_t, s_c, s_d):
    """批量流水线：同步次数压到 2（牺牲批间流水）"""
    for i in range(BATCHES):
        with torch.npu.stream(s_t):
            bufs[i % 2].copy_(hosts[i % 2], non_blocking=True)
    ev[0].record(s_t)
    s_c.wait_event(ev[0])
    outs = [None] * BATCHES
    with torch.npu.stream(s_c):
        for i in range(BATCHES):
            outs[i] = _calc(bufs[i % 2])
    ev[1].record(s_c)
    s_d.wait_event(ev[1])
    with torch.npu.stream(s_d):
        for o in outs:
            _ = o.to("cpu", non_blocking=True)


def v5_serial_same_stream(hosts, bufs, *_):
    """同流顺序执行（无重叠理论下界参照）"""
    for i in range(BATCHES):
        b = i % 2
        bufs[b].copy_(hosts[b], non_blocking=True)
        _ = _calc(bufs[b])


def run(fn, hosts, bufs, streams):
    torch.npu.synchronize()
    t0 = time.time()
    fn(hosts, bufs, *streams)
    torch.npu.synchronize()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="512,1024,2048,4096")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default="dbuf_variants_rigorous_result.json")
    args = ap.parse_args()

    _warmup_basic()
    result = {"rounds": args.rounds, "batches": BATCHES, "sizes": {}}
    print(f"=== D8 求解实验：sizes={args.sizes} rounds={args.rounds} ===")

    for n in [int(x) for x in args.sizes.split(",")]:
        hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
        bufs = [torch.zeros(n, n, device=DEV) for _ in range(2)]
        ev_h2d = [torch.npu.Event() for _ in range(2)]
        ev_calc = [torch.npu.Event() for _ in range(2)]
        s_t, s_c, s_d = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
        streams = (ev_h2d, ev_calc, s_t, s_c, s_d)
        streams_v1 = (ev_h2d, s_t, s_c, s_d)

        # 分段计时（该规模专用预热后）
        def med(f, *a):
            f(*a)  # 该规模预热（首次算子开销可达百倍）
            return statistics.median([f(*a) for _ in range(args.rounds)])

        def time_copy_sync(hosts_, bufs_):
            torch.npu.synchronize()
            t0 = time.time()
            for i in range(BATCHES):
                bufs_[i % 2].copy_(hosts_[i % 2])
            torch.npu.synchronize()
            return time.time() - t0

        def time_calc_only(bufs_):
            torch.npu.synchronize()
            t0 = time.time()
            for i in range(BATCHES):
                _calc(bufs_[i % 2])
            torch.npu.synchronize()
            return time.time() - t0

        t_copy_sync = med(time_copy_sync, hosts, bufs)
        t_calc = med(time_calc_only, bufs)

        t_v0 = med(lambda: run(v0, hosts, bufs, streams))
        t_v1 = med(lambda: run(v1, hosts, bufs, streams_v1))
        t_v4 = med(lambda: run(v4, hosts, bufs, streams_v1))
        t_v5 = med(lambda: run(v5_serial_same_stream, hosts, bufs, streams))

        base = t_copy_sync + t_calc
        row = {
            "copy_sync_ms": round(t_copy_sync * 1000, 3),
            "calc_ms": round(t_calc * 1000, 3),
            "no_overlap_baseline_ms": round(base * 1000, 3),
            "V0_ms": round(t_v0 * 1000, 3),
            "V1_ms": round(t_v1 * 1000, 3),
            "V4_ms": round(t_v4 * 1000, 3),
            "V5_serial_same_stream_ms": round(t_v5 * 1000, 3),
            "overlap_V0": round((base - t_v0) / base, 4),
            "overlap_V1": round((base - t_v1) / base, 4),
            "overlap_V4": round((base - t_v4) / base, 4),
            "sync_overhead_V0_ms": round((t_v0 - base) * 1000, 3),
            "sync_overhead_V1_ms": round((t_v1 - base) * 1000, 3),
            "sync_overhead_V4_ms": round((t_v4 - base) * 1000, 3),
        }
        result["sizes"][n] = row
        print(f"\n--- n={n} ---")
        print(f"  拷贝(同步) {row['copy_sync_ms']:.3f}ms + 计算 {row['calc_ms']:.3f}ms "
              f"= 无重叠基准 {row['no_overlap_baseline_ms']:.3f}ms")
        for k in ("V0", "V1", "V4"):
            print(f"  {k}: pipe={row[k+'_ms']:>7.3f}ms  同步开销={row['sync_overhead_'+k+'_ms']:>7.3f}ms  "
                  f"重叠率={row['overlap_'+k]:>7.1%}")
        print(f"  V5(同流无重叠参照) {row['V5_serial_same_stream_ms']:.3f}ms")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
