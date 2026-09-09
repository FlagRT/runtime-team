#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_stream_profiler_l1.py — 多流时间线 Level1（抓 CANN kernel 级数据）
═══════════════════════════════════════════════════════════════════════════════

【背景】默认 torch_npu.profiler 只采集到入队/出队瞬间事件（enqueue/dequeue）+ cpu_op，
  没有 NPU kernel 执行窗口 → 无法判定多流是否真并发。日志提示需 level1/level2。
【目标】用 ProfilerLevel.Level1 + _ExperimentalConfig 采集 CANN kernel 级时间线，
  提取非 cpu_op/enqueue/dequeue 的事件，看是否存在 kernel 执行窗口及其重叠。

【用法】容器内 A 线环境：python test_stream_profiler_l1.py
【输出】stdout + stream_profiler_l1_result.json + /tmp/stream_trace_l1.json
"""

import json
import os
from collections import Counter

import torch
import torch_npu
import torch_npu.profiler as npu_prof

TRACE = "/tmp/stream_trace_l1.json"


def pipeline(n=2048, batches=6):
    """双缓冲三流流水线：H2D → 计算 → D2H。"""
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    bufs = [torch.zeros(n, n, device="npu") for _ in range(2)]
    ev_h = [torch.npu.Event() for _ in range(2)]
    ev_c = [torch.npu.Event() for _ in range(2)]
    s_h2d = torch.npu.Stream()
    s_calc = torch.npu.Stream()
    s_d2h = torch.npu.Stream()
    for i in range(batches):
        b = i % 2
        with torch.npu.stream(s_h2d):
            bufs[b].copy_(hosts[b], non_blocking=True)
        ev_h[b].record(s_h2d)
        s_calc.wait_event(ev_h[b])
        with torch.npu.stream(s_calc):
            out = (bufs[b] @ bufs[b]).sum()
        ev_c[b].record(s_calc)
        s_d2h.wait_event(ev_c[b])
        with torch.npu.stream(s_d2h):
            out.cpu()
    torch.npu.synchronize()


def main():
    print("=== test_stream_profiler_l1.py: CANN kernel 级时间线 ===")
    torch.zeros(1, device="npu")
    pipeline(n=512, batches=2)  # warmup（首次算子开销 107 倍）

    exp_cfg = npu_prof._ExperimentalConfig(
        profiler_level=npu_prof.ProfilerLevel.Level1)
    with npu_prof.profile(
            activities=[npu_prof.ProfilerActivity.CPU, npu_prof.ProfilerActivity.NPU],
            experimental_config=exp_cfg, record_shapes=False) as prof:
        pipeline(n=2048, batches=6)
    prof.export_chrome_trace(TRACE)
    print(f"[trace] 导出 {TRACE} ({os.path.getsize(TRACE)//1024} KB)")

    with open(TRACE) as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else (
        data if isinstance(data, list) else [])
    xev = [e for e in events if e.get("ph") == "X"]
    cats = Counter(e.get("cat", "?") for e in xev)
    print(f"[trace] cat 分布: {cats.most_common(8)}")

    # 非标准类别 = 可能的 kernel/设备执行窗口
    others = [e for e in xev if e.get("cat") not in ("cpu_op", "enqueue", "dequeue")]
    print(f"[trace] 非标准类别事件数: {len(others)}")
    for e in others[:8]:
        print(f"   cat={e.get('cat')} name={str(e.get('name'))[:45]} dur={e.get('dur')}")

    # 若有 kernel 窗口，按时间排序看重叠（不同 pid/tid 上）
    kernel_like = []
    for e in xev:
        cat = str(e.get("cat", ""))
        name = str(e.get("name", ""))
        if cat in ("cpu_op", "enqueue", "dequeue"):
            continue
        try:
            kernel_like.append({"name": name, "cat": cat,
                                "ts": float(e.get("ts", 0)), "dur": float(e.get("dur", 0)),
                                "tid": e.get("tid")})
        except (TypeError, ValueError):
            continue
    overlaps = 0
    for i in range(len(kernel_like)):
        for j in range(i + 1, len(kernel_like)):
            a, b = kernel_like[i], kernel_like[j]
            ov = min(a["ts"] + a["dur"], b["ts"] + b["dur"]) - max(a["ts"], b["ts"])
            if ov > 0:
                overlaps += 1
    print(f"[分析] kernel 类事件={len(kernel_like)} 重叠对={overlaps}")

    res = {"trace_path": TRACE, "trace_kb": os.path.getsize(TRACE) // 1024,
           "cat_distribution": dict(cats), "nonstandard_events": len(others),
           "kernel_like_events": len(kernel_like), "kernel_overlap_pairs": overlaps,
           "sample_nonstandard": [{"cat": e.get("cat"), "name": str(e.get("name"))[:60],
                                   "dur": e.get("dur")} for e in others[:10]]}
    verdict = ("KERNEL_TIMELINE_CAPTURED" if len(kernel_like) > 0
               else "NO_KERNEL_EVENTS")
    res["verdict"] = verdict
    res["note"] = ("已采集到 kernel 级设备事件，可判定多流并发" if len(kernel_like) > 0
                   else "Level1 仍未采集到 kernel 执行窗口 → 需 msprof 或 Level2/AiCMetrics")
    print(f"\n{verdict}: {res['note']}")

    with open("stream_profiler_l1_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入 stream_profiler_l1_result.json")


if __name__ == "__main__":
    main()
