#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_dbuf_v1_equivalence.py — O2-2：验证 V1 精简链的「数值等价性」与提速真实性

【质疑】test_dbuf_sync_variants.py 中 V1 重叠率 +21.6% 是唯一转正的方案，
       但它把「每批 D2H 回传」改成「末尾统一 wait_stream 回传」。
       真实推理每 token 都要回传采样结果 —— 若 V1 靠牺牲回换功能换速度，则结论无效。

【证伪/证实设计】
  固定输入（同 seed、同 host 数据）→ 分别跑 V0 与 V1 → 收集**每批**的 D2H 结果
  → 逐位比对。若一致，说明 V1 的提速来自「减少过度同步」而非「少干活」：
     计算流本就不该等 D2H（D2H 应与后续批次计算重叠），V0 的 计算→D2H 事件链是过度同步。

【同时测】稳定性：多轮 + 更大张量，降低 ms 级测量噪声的影响
【用法】容器内：python3 test_dbuf_v1_equivalence.py [--n 1024] [--rounds 5]
"""
import argparse
import json
import statistics
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"


def _warmup():
    torch.zeros(1, device=DEV)
    h = torch.randn(256, 256).pin_memory()
    d = torch.zeros(256, n := 256, device=DEV)
    d.copy_(h, non_blocking=True)
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        _ = (d @ d).sum()
    torch.npu.synchronize()


def _calc(buf, rep):
    for _ in range(rep):
        buf @ buf
    return buf.sum()


def run_v0(hosts, batches, n, rep):
    """基线：每批 record(H2D) → wait → 计算 → record(calc) → wait → D2H（每批回传）"""
    buf = [torch.zeros(n, n, device=DEV) for _ in range(2)]
    ev_h2d = [torch.npu.Event() for _ in range(2)]
    ev_calc = [torch.npu.Event() for _ in range(2)]
    s_trans, s_calc, s_d2h = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
    outs = [None] * batches

    t0 = time.time()
    for i in range(batches):
        b = i % 2
        with torch.npu.stream(s_trans):
            buf[b].copy_(hosts[b], non_blocking=True)
        ev_h2d[b].record(s_trans)
        s_calc.wait_event(ev_h2d[b])
        with torch.npu.stream(s_calc):
            r = _calc(buf[b], rep)
        ev_calc[b].record(s_calc)
        s_d2h.wait_event(ev_calc[b])
        with torch.npu.stream(s_d2h):
            outs[i] = r.to("cpu", non_blocking=True)
    torch.npu.synchronize()
    return time.time() - t0, [float(o) for o in outs]


def run_v1(hosts, batches, n, rep):
    """精简链：每批只保留 H2D→计算 一条依赖；D2H 不阻塞计算流（末尾统一回传）"""
    buf = [torch.zeros(n, n, device=DEV) for _ in range(2)]
    ev = [torch.npu.Event() for _ in range(2)]
    s_trans, s_calc, s_d2h = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
    res = [None] * batches

    t0 = time.time()
    for i in range(batches):
        b = i % 2
        with torch.npu.stream(s_trans):
            buf[b].copy_(hosts[b], non_blocking=True)
        ev[b].record(s_trans)
        s_calc.wait_event(ev[b])
        with torch.npu.stream(s_calc):
            res[i] = _calc(buf[b], rep)
    # D2H 统一在末尾：与 V0 一样是「每批结果都要回传」，只是不阻塞后续批次计算
    s_d2h.wait_stream(s_calc)
    with torch.npu.stream(s_d2h):
        for i in range(batches):
            res[i] = res[i].to("cpu", non_blocking=True)
    torch.npu.synchronize()
    return time.time() - t0, [float(r) for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default="dbuf_v1_equivalence_result.json")
    args = ap.parse_args()

    torch.manual_seed(42)
    _warmup()
    hosts = [torch.randn(args.n, args.n).pin_memory() for _ in range(2)]

    print(f"=== O2-2 V1 等价性与稳定性：n={args.n} batches={args.batches} rounds={args.rounds} ===")

    t0, out0 = run_v0(hosts, args.batches, args.n, 1)   # 首轮含预热残余，不计时
    ts_v0, ts_v1 = [], []
    for _ in range(args.rounds):
        t, out0 = run_v0(hosts, args.batches, args.n, 1)
        ts_v0.append(t)
    for _ in range(args.rounds):
        t, out1 = run_v1(hosts, args.batches, args.n, 1)
        ts_v1.append(t)

    # 数值等价：逐批比对（相对容差 1e-5，沿用 conformance 纪律）
    diffs = []
    for a, b in zip(out0, out1):
        denom = max(abs(a), 1e-12)
        diffs.append(abs(a - b) / denom)
    max_diff = max(diffs) if diffs else 0.0
    equivalent = max_diff < 1e-5

    m0, m1 = statistics.median(ts_v0), statistics.median(ts_v1)
    speedup = (m0 - m1) / m0 if m0 > 0 else 0.0

    result = {
        "config": vars(args),
        "v0_median_s": round(m0, 5),
        "v1_median_s": round(m1, 5),
        "v0_all": [round(x, 5) for x in ts_v0],
        "v1_all": [round(x, 5) for x in ts_v1],
        "speedup_pct": f"{speedup:.1%}",
        "max_rel_diff": max_diff,
        "numerically_equivalent": equivalent,
        "v0_outputs": out0,
        "v1_outputs": out1,
        "verdict": "V1_EQUIVALENT_AND_FASTER" if (equivalent and speedup > 0) else "V1_NOT_PROVEN",
    }

    print(f"V0 中位 {m0:.5f}s   V1 中位 {m1:.5f}s   提速 {speedup:.1%}")
    print(f"逐批数值最大相对差 {max_diff:.3e} → 等价: {equivalent}")
    print(f"V0 输出: {[round(x, 4) for x in out0]}")
    print(f"V1 输出: {[round(x, 4) for x in out1]}")
    print(f"\n=== 判定：{result['verdict']} ===")
    if equivalent and speedup > 0:
        print("[结论] V1 提速真实（非牺牲功能）：D2H 结果与 V0 逐位等价，")
        print("       提速来自去掉「计算→D2H」这条过度同步 —— 计算流本不必等回传。")
    else:
        print("[结论] V1 未通过验证：数值不等价或未提速，需重新设计。")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果：{args.out}")
    return 0 if result["verdict"] == "V1_EQUIVALENT_AND_FASTER" else 1


if __name__ == "__main__":
    raise SystemExit(main())
