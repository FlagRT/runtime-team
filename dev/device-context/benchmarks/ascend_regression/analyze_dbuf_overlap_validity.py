#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_dbuf_overlap_validity.py — 检验「n≤1024 重叠率为负 = 物理限制」这一结论是否成立

【为什么要做】前一轮结论依据不足，存在三处方法缺陷：
  1. 重叠率分母不可比：serial 基线用**同步**拷贝、pipe 用**异步**拷贝，混淆了
     「流水线重叠」与「拷贝方式差异」
  2. 只测 3 个规模点，且未分离拷贝时间与计算时间，无法判断是"计算太小"还是"拷贝占比变化"
  3. **自身数据即反证**：V3（计算×8）在 n=2048 时重叠率 15.5%，远低于 V0 的 31.0%。
     若真是"计算量摊薄同步开销"，计算量最大的 V3 应最高 —— 原结论不成立

【本脚本做什么】分段计时，把时间构成拆开，用三种口径重算重叠率：
  t_copy_sync  同步拷贝 6 批（serial 基线所用方式）
  t_copy_async 异步拷贝 6 批（含等待，pipe 所用方式）
  t_calc       纯计算 6 批（无传输）
  t_pipe       V0 流水线墙钟
  t_serial     当前 serial 基线墙钟（同步拷贝 + 同流计算）

  overlap_current = (t_serial - t_pipe) / t_serial           —— 当前指标（分母不可比）
  overlap_ideal   = (t_copy_sync + t_calc - t_pipe) / (...)  —— 理论无重叠作分母
  overlap_ceiling = 1 - max(t_copy_async, t_calc)/(两者之和)  —— 理论上限（完美重叠时的最好情况）

【判定】若某规模下 t_pipe 已接近 overlap_ceiling 对应时间，说明实现没问题、是负载结构所致；
       若 t_pipe 远高于理论下限，说明仍有实现层优化空间，不能归为"物理限制"。

【用法】容器内：python3 analyze_dbuf_overlap_validity.py [--sizes 256,512,1024,2048] [--rounds 7]
"""
import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"
BATCHES = 6


def _warmup():
    torch.zeros(1, device=DEV)
    h = torch.randn(256, 256).pin_memory()
    d = torch.zeros(256, 256, device=DEV)
    d.copy_(h, non_blocking=True)
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        _ = (d @ d).sum()
    torch.npu.synchronize()


def time_copy_sync(hosts, bufs):
    """同步拷贝（serial 基线所用）"""
    torch.npu.synchronize()
    t0 = time.time()
    for i in range(BATCHES):
        bufs[i % 2].copy_(hosts[i % 2])
    torch.npu.synchronize()
    return time.time() - t0


def time_copy_async(hosts, bufs):
    """异步拷贝 + 等待（pipe 所用），含 pin_memory non_blocking 路径开销"""
    s = torch.npu.Stream()
    torch.npu.synchronize()
    t0 = time.time()
    with torch.npu.stream(s):
        for i in range(BATCHES):
            bufs[i % 2].copy_(hosts[i % 2], non_blocking=True)
    torch.npu.synchronize()
    return time.time() - t0


def time_calc(bufs):
    """纯计算 6 批（无传输）"""
    torch.npu.synchronize()
    t0 = time.time()
    for i in range(BATCHES):
        bufs[i % 2] @ bufs[i % 2]
    torch.npu.synchronize()
    return time.time() - t0


def time_pipe_v0(hosts, bufs, ev_h2d, ev_calc, s_trans, s_calc, s_d2h):
    """V0 流水线墙钟"""
    torch.npu.synchronize()
    t0 = time.time()
    for i in range(BATCHES):
        b = i % 2
        with torch.npu.stream(s_trans):
            bufs[b].copy_(hosts[b], non_blocking=True)
        ev_h2d[b].record(s_trans)
        s_calc.wait_event(ev_h2d[b])
        with torch.npu.stream(s_calc):
            r = bufs[b] @ bufs[b]
        ev_calc[b].record(s_calc)
        s_d2h.wait_event(ev_calc[b])
        with torch.npu.stream(s_d2h):
            _ = r.to("cpu", non_blocking=True)
    torch.npu.synchronize()
    return time.time() - t0


def time_serial_current(hosts, bufs):
    """当前 serial 基线：同流 同步拷贝 + 计算"""
    torch.npu.synchronize()
    t0 = time.time()
    for i in range(BATCHES):
        b = i % 2
        bufs[b].copy_(hosts[b])
        _ = bufs[b] @ bufs[b]
    torch.npu.synchronize()
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="256,512,1024,1536,2048")
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default="dbuf_overlap_validity_result.json")
    args = ap.parse_args()

    _warmup()
    sizes = [int(s) for s in args.sizes.split(",")]
    result = {"rounds": args.rounds, "batches": BATCHES, "sizes": {}, "note": ""}

    print(f"=== 重叠率有效性检验：sizes={sizes} rounds={args.rounds} ===")
    print(f"{'n':>6}{'copy_sync':>11}{'copy_async':>11}{'calc':>10}{'pipe':>10}"
          f"{'serial':>10}{'ovl_cur':>9}{'ovl_ideal':>10}{'ceiling':>9}")

    for n in sizes:
        hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
        bufs = [torch.zeros(n, n, device=DEV) for _ in range(2)]
        ev_h2d = [torch.npu.Event() for _ in range(2)]
        ev_calc = [torch.npu.Event() for _ in range(2)]
        s_trans = torch.npu.Stream()
        s_calc = torch.npu.Stream()
        s_d2h = torch.npu.Stream()

        # 预热该规模（首次算子开销）
        time_pipe_v0(hosts, bufs, ev_h2d, ev_calc, s_trans, s_calc, s_d2h)

        def med(fn, *a):
            return statistics.median([fn(*a) for _ in range(args.rounds)])

        t_copy_sync = med(time_copy_sync, hosts, bufs)
        t_copy_async = med(time_copy_async, hosts, bufs)
        t_calc = med(time_calc, bufs)
        t_pipe = med(time_pipe_v0, hosts, bufs, ev_h2d, ev_calc, s_trans, s_calc, s_d2h)
        t_serial = med(time_serial_current, hosts, bufs)

        ideal_serial = t_copy_sync + t_calc
        ovl_cur = (t_serial - t_pipe) / t_serial if t_serial > 0 else 0.0
        ovl_ideal = (ideal_serial - t_pipe) / ideal_serial if ideal_serial > 0 else 0.0
        # 理论上限：完美重叠时总时间 = max(拷贝, 计算)，故相对"无重叠"的最大节省比例
        denom = t_copy_async + t_calc
        ceiling = 1 - (max(t_copy_async, t_calc) / denom) if denom > 0 else 0.0

        result["sizes"][n] = {
            "copy_sync_ms": round(t_copy_sync * 1000, 3),
            "copy_async_ms": round(t_copy_async * 1000, 3),
            "calc_ms": round(t_calc * 1000, 3),
            "pipe_ms": round(t_pipe * 1000, 3),
            "serial_ms": round(t_serial * 1000, 3),
            "ideal_serial_ms": round(ideal_serial * 1000, 3),
            "overlap_current": round(ovl_cur, 4),
            "overlap_ideal": round(ovl_ideal, 4),
            "overlap_ceiling": round(ceiling, 4),
            "achieved_vs_ceiling": round(ovl_ideal / ceiling, 3) if ceiling > 0 else None,
        }
        print(f"{n:>6}{t_copy_sync*1000:>11.3f}{t_copy_async*1000:>11.3f}{t_calc*1000:>10.3f}"
              f"{t_pipe*1000:>10.3f}{t_serial*1000:>10.3f}{ovl_cur:>8.1%}{ovl_ideal:>10.1%}{ceiling:>9.1%}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果：{args.out}")
    print("说明：ovl_ideal 用『同步拷贝+纯计算』作无重叠基准；ceiling = 完美重叠的理论上限。")
    print("      achieved_vs_ceiling 越接近 1 说明实现越接近理论最好；远小于 1 说明仍有实现层空间。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
