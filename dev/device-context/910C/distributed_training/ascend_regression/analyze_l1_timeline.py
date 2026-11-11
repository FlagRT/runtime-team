#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
analyze_l1_timeline.py — 从 Level1 kernel trace 精确判定"多流是否真并发"
═══════════════════════════════════════════════════════════════════════════════

【输入】/tmp/stream_trace_l1.json（torch_npu.profiler Level1 导出的 kernel 级 trace）
【方法】
  1. 提取 kernel 事件（cat=None 的设备执行事件），按类型分类：
     copy(H2D/D2H memcpy) / compute(Matmul/ReduceSum) / zero / event(EVENT_RECORD/WAIT)
  2. 时间线排序，计算 copy 与 compute 的时间重叠窗口（真实并发证据）
  3. 统计 event 等待开销（EVENT_WAIT 的 dur）
【判据】
  - copy 与 compute 存在 >0 重叠 → 多流并发成立（H2D DMA 与 AI Core 计算并行）
  - 若 copy 与 compute 完全串行（零重叠）→ CANN 未并发调度多流
【输出】stdout + kernel_timeline_analysis.json
"""

import json


def load_events(path):
    with open(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else (
        data if isinstance(data, list) else [])
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") in ("cpu_op", "enqueue", "dequeue"):
            continue
        try:
            out.append({"name": str(e.get("name", "")), "ts": float(e.get("ts", 0)),
                        "dur": float(e.get("dur", 0))})
        except (TypeError, ValueError):
            continue
    return out


def classify(name):
    n = name.lower()
    if "memcpy" in n or "memset" in n:
        return "copy"
    if "matmul" in n or "matmulv2" in n or "gemm" in n:
        return "compute_matmul"
    if "reducesum" in n:
        return "compute_reduce"
    if "zeroslike" in n or "inplacezero" in n:
        return "zero"
    if "event_record" in n or "event_wait" in n:
        return "event"
    return "other"


def main():
    path = "/tmp/stream_trace_l1.json"
    ev = load_events(path)
    print(f"=== kernel 事件 {len(ev)} 个 ===")
    from collections import Counter, defaultdict
    cats = Counter(classify(e["name"]) for e in ev)
    print(f"分类: {dict(cats)}")

    by = defaultdict(list)
    for e in ev:
        by[classify(e["name"])].append(e)

    # 1. copy vs compute(matmul) 重叠
    copies = by.get("copy", [])
    matmuls = by.get("compute_matmul", [])
    overlaps = []
    for c in copies:
        cs, ce = c["ts"], c["ts"] + c["dur"]
        for m in matmuls:
            ms, me = m["ts"], m["ts"] + m["dur"]
            ov = min(ce, me) - max(cs, ms)
            if ov > 0:
                overlaps.append({"copy_dur": round(c["dur"], 2),
                                 "matmul_dur": round(m["dur"], 2),
                                 "overlap_us": round(ov, 2)})
    print(f"\n[判据] copy={len(copies)} matmul={len(matmuls)} → copy∩matmul 重叠对={len(overlaps)}")
    for o in overlaps[:5]:
        print(f"   copy({o['copy_dur']}us) ∥ matmul({o['matmul_dur']}us) 重叠 {o['overlap_us']}us")

    # 2. event 等待开销
    evs = by.get("event", [])
    waits = [e for e in evs if "wait" in e["name"].lower()]
    if waits:
        tot = sum(e["dur"] for e in waits)
        print(f"\n[同步开销] EVENT_WAIT n={len(waits)} 累计={tot:.1f}us "
              f"最大={max(e['dur'] for e in waits):.1f}us")

    # 3. 时间跨度与设备利用率
    if ev:
        t0 = min(e["ts"] for e in ev)
        t1 = max(e["ts"] + e["dur"] for e in ev)
        busy = sum(e["dur"] for e in ev)
        print(f"\n[时间线] 跨度 {(t1-t0)/1000:.2f}ms, kernel 累计 {busy/1000:.2f}ms "
              f"(含重叠，故可 >100%)")

    verdict = "MULTI_STREAM_CONCURRENT" if overlaps else "NO_COPY_COMPUTE_OVERLAP"
    note = (f"kernel 时间线证实：H2D copy 与 matmul 存在 {len(overlaps)} 处时间重叠 → "
            f"昇腾多流（DMA 传输 ∥ AI Core 计算）**真并发成立**" if overlaps else
            "copy 与 compute 无时间重叠 → CANN 未并发调度多流")
    print(f"\n{verdict}: {note}")

    res = {"verdict": verdict, "note": note, "n_events": len(ev), "cats": dict(cats),
           "copy_matmul_overlaps": len(overlaps), "overlap_samples": overlaps[:10],
           "wait_total_us": round(sum(e["dur"] for e in waits), 2) if waits else 0,
           "wait_max_us": round(max((e["dur"] for e in waits), default=0), 2)}
    with open("kernel_timeline_analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入 kernel_timeline_analysis.json")


if __name__ == "__main__":
    main()
