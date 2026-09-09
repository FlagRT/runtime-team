#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_stream_timeline_profiler.py — 多流时间线抓取（A4 决定性证据）
═══════════════════════════════════════════════════════════════════════════════

【目标】回答"昇腾多流能否真并发"——用 torch.profiler 抓 NPU(PrivateUse1) kernel 时间线，
  判定 H2D(copy) 与 计算(matmul) 是否在时间轴上重叠，进而决定 D8 双缓冲职责结论：
  多流流水线 vs 单流批量优化。

【方法】
  1. warmup（规避首次算子开销 107 倍，见 breakdown S4/S5）
  2. torch.profiler(activities=[CPU, PrivateUse1]) 包裹双缓冲流水线
  3. 导出 chrome trace（含每个 kernel 的 ts/dur/stream）
  4. 解析 trace：提取 device 事件，按 (name, ts, dur) 重建时间线，判定 copy 与 matmul 重叠

【用法】容器内 A 线环境：python test_stream_timeline_profiler.py
【输出】stdout + stream_timeline_result.json + /tmp/stream_trace.json（chrome trace）
"""

import json
import os

import torch
import torch_npu
from torch.profiler import ProfilerActivity, profile

TRACE_OUT = "/tmp/stream_trace.json"


def _pipeline(n=2048, batches=6):
    """双缓冲流水线：H2D(传输流) → 计算(计算流) → D2H(回传流)。"""
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    bufs = [torch.zeros(n, n, device="npu") for _ in range(2)]
    ev_h = [torch.npu.Event() for _ in range(2)]
    ev_c = [torch.npu.Event() for _ in range(2)]
    st, sc, sd = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
    for i in range(batches):
        b = i % 2
        with torch.npu.stream(st):
            bufs[b].copy_(hosts[b], non_blocking=True)   # H2D
        ev_h[b].record(st)
        sc.wait_event(ev_h[b])
        with torch.npu.stream(sc):
            out = (bufs[b] @ bufs[b]).sum()              # 计算
        ev_c[b].record(sc)
        sd.wait_event(ev_c[b])
        with torch.npu.stream(sd):
            out.cpu()                                     # D2H
    torch.npu.synchronize()


def _analyze_trace(path):
    """从 chrome trace 提取 device 事件，判定 copy 与 matmul 时间重叠。"""
    with open(path) as f:
        data = json.load(f)
    # 昇腾 torch_npu.profiler 导出为 list；torch.profiler 为 {"traceEvents": [...]}
    if isinstance(data, dict):
        events = data.get("traceEvents", [])
    elif isinstance(data, list):
        events = data
    else:
        events = []
    dev = []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat", "")
        name = e.get("name", "")
        # device 侧 kernel/拷贝事件（排除 CPU 侧的 aten 算子）
        if any(k in cat.lower() for k in ("kernel", "gpu", "device", "npu", "acl", "memcpy")) or \
           any(k in name.lower() for k in ("memcpy", "matmul", "acl", "gemm", "memset")):
            try:
                ts = float(e.get("ts", 0))
                dur = float(e.get("dur", 0))
            except (TypeError, ValueError):
                continue
            dev.append({
                "name": name,
                "ts": ts,
                "dur": dur,
                "cat": cat,
                "tid": e.get("tid"),
                "pid": e.get("pid"),
            })
    return dev


def _cat_summary(path):
    """打印 trace 中所有事件类别分布（确认是否存在 NPU/kernel 侧事件）。"""
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else (
        data if isinstance(data, list) else [])
    from collections import Counter
    return Counter(e.get("cat", "?") for e in events if e.get("ph") == "X")


def _overlap_analysis(dev):
    """按时间区间判定 copy 类与 matmul 类事件是否重叠。"""
    def cls(n):
        n = n.lower()
        if "memcpy" in n or "memset" in n or "copy" in n:
            return "copy"
        if "matmul" in n or "gemm" in n or "mm" in n:
            return "compute"
        return "other"

    copies = [d for d in dev if cls(d["name"]) == "copy"]
    computes = [d for d in dev if cls(d["name"]) == "compute"]
    overlaps = 0
    detail = []
    for c in copies:
        cs, ce = c["ts"], c["ts"] + c["dur"]
        for m in computes:
            ms, me = m["ts"], m["ts"] + m["dur"]
            ov = min(ce, me) - max(cs, ms)
            if ov > 0:
                overlaps += 1
                detail.append({"copy": c["name"], "compute": m["name"], "overlap_us": ov})
    return {"n_copy": len(copies), "n_compute": len(computes),
            "n_overlap_pairs": overlaps, "sample": detail[:5]}


def main():
    print("=== test_stream_timeline_profiler.py: 多流时间线（A4 决定性证据）===")
    torch.zeros(1, device="npu")
    _pipeline(n=512, batches=2)    # warmup（规避首次算子 107 倍开销）

    # 优先用昇腾专用 torch_npu.profiler（ProfilerActivity.NPU 才抓 device kernel 时间线；
    # torch.profiler 的 PrivateUse1 在本环境只记录 cpu_op，无法判定设备并发）
    prof = None
    try:
        import torch_npu.profiler as npu_prof
        with npu_prof.profile(
                activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
                record_shapes=False, with_stack=False) as prof:
            _pipeline(n=2048, batches=6)
        print("[profiler] 使用 torch_npu.profiler（NPU activity）")
    except Exception as e:
        print(f"[profiler] torch_npu.profiler 失败({e})，退回 torch.profiler（仅 cpu_op）")
        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
                         record_shapes=False, with_stack=False) as prof:
                _pipeline(n=2048, batches=6)
        except Exception as e2:
            print(f"PROFILER_FAIL: {e2}")
            return

    # 汇总（昇腾 torch_npu.profiler 无 key_averages，容错跳过）
    rows = []
    try:
        for e in prof.key_averages():
            dt = getattr(e, "device_time_total", 0) or 0
            if dt > 0:
                rows.append((e.key, dt, e.count))
        rows.sort(key=lambda r: -r[1])
        print("\n=== key_averages（device time top 10）===")
        for k, dt, c in rows[:10]:
            print(f"  {k}: device={dt}us count={c}")
    except AttributeError:
        print("\n[profiler] 昇腾 profiler 无 key_averages()，直接分析 trace")

    # 导出 chrome trace 并分析
    try:
        prof.export_chrome_trace(TRACE_OUT)
        print(f"\n[trace] 已导出 {TRACE_OUT} ({os.path.getsize(TRACE_OUT)/1024:.1f} KB)")
    except Exception as e:
        print(f"TRACE_EXPORT_FAIL: {e}")
        return

    cats = _cat_summary(TRACE_OUT)
    print(f"[trace] 事件类别分布: {dict(cats)}")
    dev = _analyze_trace(TRACE_OUT)
    print(f"[trace] device 事件数={len(dev)}")
    for d in dev[:8]:
        print(f"  ts={d['ts']} dur={d['dur']}us {d['name'][:50]} (cat={d['cat']})")

    oa = _overlap_analysis(dev)
    print(f"\n[重叠分析] copy={oa['n_copy']} compute={oa['n_compute']} "
          f"重叠对={oa['n_overlap_pairs']}")

    verdict = ("MULTI_STREAM_CONCURRENT" if oa["n_overlap_pairs"] > 0
               else "NO_DEVICE_OVERLAP")
    note = ("device 时间线上 copy 与 compute 存在重叠 → 多流真并发成立"
            if oa["n_overlap_pairs"] > 0 else
            "device 时间线上未捕获 copy/compute 重叠 → 需结合 trace 人工确认（或 profiler 未采集 NPU kernel）")
    res = {"verdict": verdict, "note": note, "overlap_analysis": oa,
           "device_events": len(dev), "trace_path": TRACE_OUT,
           "key_averages_top": [{"name": k, "device_us": dt, "count": c} for k, dt, c in rows[:10]]}
    print(f"\n{verdict}: {note}")

    with open("stream_timeline_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入 stream_timeline_result.json")


if __name__ == "__main__":
    main()
