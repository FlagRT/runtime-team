#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_double_buffer_breakdown.py — 双缓冲重叠未发生的分段计时定位探针（A4）
═══════════════════════════════════════════════════════════════════════════════

【问题】test_double_buffer_pipeline.py 结果 DBUF2_PARTIAL：
  数据一致 ✅，但 pipe 0.074s vs serial 0.005s（慢 14 倍，重叠率 -1306%）。
【目标】定位"同步退化" vs "流切换开销" vs "事件链开销"：
  S1 纯传输（non_blocking=True，当前流）：**发起耗时 vs 完成耗时** → 判定异步性
  S2 纯传输（non_blocking=False）：同步基线对照
  S3 纯传输（专用流 + non_blocking）：流切换影响
  S4/S5 纯计算（当前流 / 专用流）：流切换对计算的影响
  S6 事件链（record+wait 空负载）：单次事件开销
  S7 完整流水线（512²）：复现基线
  S8 完整流水线（2048²）：大张量（计算 >> 同步开销）时重叠是否出现

【判据】
  - S1 发起耗时 ≈ 完成耗时 → copy_ non_blocking **退化为同步**（torch_npu 缺口）
  - S1 发起耗时 << 完成耗时 → 异步正常，问题在别处（看 S3/S6 开销）
  - S8 重叠率 > 0 而 S7 < 0 → **同步开销主导**，增大计算粒度可掩盖
  - S6 单次事件开销 ~ms 级 → **事件链开销主导**

【用法】容器内 A 线环境：python test_double_buffer_breakdown.py
【输出】stdout + double_buffer_breakdown_result.json
"""

import json
import time

import torch
import torch_npu


def _t():
    return time.perf_counter()


def main():
    print("=== test_double_buffer_breakdown.py: 双缓冲重叠分段定位 ===")
    torch.zeros(1, device="npu")  # 预热
    n_batches = 6
    res = {"checks": {}, "timing": {}, "note": ""}

    # ── S1: 纯传输（non_blocking=True，当前流）：拆"发起"与"完成" ──
    n = 512
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = torch.zeros(n, n, device="npu")
    t0 = _t()
    for i in range(n_batches):
        buf.copy_(hosts[i % 2], non_blocking=True)
    t_issue = _t() - t0          # 入队耗时（若真异步 → 极短）
    torch.npu.synchronize()
    t_total = _t() - t0          # 含实际传输完成
    res["timing"]["s1_h2d_async_issue_s"] = round(t_issue, 6)
    res["timing"]["s1_h2d_async_total_s"] = round(t_total, 6)
    async_ratio = t_issue / t_total if t_total > 0 else 1.0
    res["checks"]["s1_h2d_async"] = {
        "ok": async_ratio < 0.5,
        "detail": f"non_blocking 入队={t_issue*1e3:.2f}ms 完成={t_total*1e3:.2f}ms 入队占比={async_ratio:.1%}"
                  f" → {'异步正常' if async_ratio < 0.5 else '疑似退化为同步'}"}
    print(f"[S1] H2D non_blocking: 入队 {t_issue*1e3:.2f}ms / 完成 {t_total*1e3:.2f}ms "
          f"(入队占比 {async_ratio:.1%})")

    # ── S2: 纯传输（non_blocking=False，同步基线）──
    t0 = _t()
    for i in range(n_batches):
        buf.copy_(hosts[i % 2], non_blocking=False)
    torch.npu.synchronize()
    t_sync_copy = _t() - t0
    res["timing"]["s2_h2d_sync_s"] = round(t_sync_copy, 6)
    print(f"[S2] H2D 同步拷贝基线: {t_sync_copy*1e3:.2f}ms")

    # ── S3: 纯传输（专用流 + non_blocking）──
    s_trans = torch.npu.Stream()
    t0 = _t()
    with torch.npu.stream(s_trans):
        for i in range(n_batches):
            buf.copy_(hosts[i % 2], non_blocking=True)
    t_issue_s = _t() - t0
    torch.npu.synchronize()
    t_total_s = _t() - t0
    res["timing"]["s3_h2d_stream_issue_s"] = round(t_issue_s, 6)
    res["timing"]["s3_h2d_stream_total_s"] = round(t_total_s, 6)
    stream_overhead = t_total_s - t_total
    print(f"[S3] H2D 专用流: 入队 {t_issue_s*1e3:.2f}ms / 完成 {t_total_s*1e3:.2f}ms "
          f"(相对当前流 +{stream_overhead*1e3:.2f}ms)")

    # ── S4/S5: 纯计算（当前流 / 专用流）──
    d = torch.randn(n, n, device="npu")
    t0 = _t()
    for _ in range(n_batches):
        (d @ d).sum()
    torch.npu.synchronize()
    t_calc_cur = _t() - t0
    s_calc = torch.npu.Stream()
    t0 = _t()
    with torch.npu.stream(s_calc):
        for _ in range(n_batches):
            (d @ d).sum()
    torch.npu.synchronize()
    t_calc_s = _t() - t0
    res["timing"]["s4_calc_current_s"] = round(t_calc_cur, 6)
    res["timing"]["s5_calc_stream_s"] = round(t_calc_s, 6)
    print(f"[S4] 计算(当前流) {t_calc_cur*1e3:.2f}ms | [S5] 计算(专用流) {t_calc_s*1e3:.2f}ms "
          f"(差 {abs(t_calc_s-t_calc_cur)*1e3:.2f}ms)")

    # ── S6: 事件链开销（record + wait 空负载）──
    ev = torch.npu.Event()
    t0 = _t()
    for _ in range(n_batches):
        ev.record()
        torch.npu.current_stream().wait_event(ev)
    torch.npu.synchronize()
    t_event = _t() - t0
    per_event = t_event / n_batches
    res["timing"]["s6_event_chain_s"] = round(t_event, 6)
    res["timing"]["s6_per_event_ms"] = round(per_event * 1e3, 3)
    print(f"[S6] 事件链 6×(record+wait): {t_event*1e3:.2f}ms → 单次 {per_event*1e3:.3f}ms")

    # ── S7/S8: 完整流水线（512 / 2048）──
    def run_pipeline(n):
        hs = [torch.randn(n, n).pin_memory() for _ in range(2)]
        bufs = [torch.zeros(n, n, device="npu") for _ in range(2)]
        ev_h = [torch.npu.Event() for _ in range(2)]
        ev_c = [torch.npu.Event() for _ in range(2)]
        st, sc, sd = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
        cur = torch.npu.current_stream()
        t0 = _t()
        for i in range(n_batches):
            b = i % 2
            with torch.npu.stream(st):
                bufs[b].copy_(hs[b], non_blocking=True)
            ev_h[b].record(st)
            sc.wait_event(ev_h[b])
            with torch.npu.stream(sc):
                out = (bufs[b] @ bufs[b]).sum()
            ev_c[b].record(sc)
            sd.wait_event(ev_c[b])
            with torch.npu.stream(sd):
                out_cpu = out.cpu()
        torch.npu.synchronize()
        pipe = _t() - t0
        # 串行参考
        t1 = _t()
        for i in range(n_batches):
            dd = hs[i % 2].to("npu")
            (dd @ dd).sum().cpu()
        torch.npu.synchronize()
        serial = _t() - t1
        overlap = (serial - pipe) / serial if serial > 0 else 0
        return pipe, serial, overlap, out_cpu

    for size, key in ((512, "s7"), (2048, "s8")):
        pipe, serial, overlap, out_cpu = run_pipeline(size)
        res["timing"][f"{key}_pipe_{size}_s"] = round(pipe, 6)
        res["timing"][f"{key}_serial_{size}_s"] = round(serial, 6)
        res["timing"][f"{key}_overlap_{size}"] = round(overlap, 4)
        print(f"[{key.upper()}] {size}²: pipe={pipe*1e3:.2f}ms serial={serial*1e3:.2f}ms 重叠率={overlap:.1%}")

    # ── S9: 真实推理负载尺度复测（4096² × 10 批，先 warmup 排除首次算子开销）
    #       判据：若重叠转正 → 证实"事件链固定开销被大计算掩盖"，流水线在真实负载有效
    def run_pipeline_n(n, batches, warmup=True):
        hs = [torch.randn(n, n).pin_memory() for _ in range(2)]
        bufs = [torch.zeros(n, n, device="npu") for _ in range(2)]
        ev_h = [torch.npu.Event() for _ in range(2)]
        ev_c = [torch.npu.Event() for _ in range(2)]
        st, sc, sd = torch.npu.Stream(), torch.npu.Stream(), torch.npu.Stream()
        cur = torch.npu.current_stream()
        if warmup:                       # 预热：首次 matmul/拷贝含算子初始化（坑 A1）
            with torch.npu.stream(st):
                bufs[0].copy_(hs[0], non_blocking=True)
            with torch.npu.stream(sc):
                (bufs[0] @ bufs[0]).sum()
            torch.npu.synchronize()
        t0 = _t()
        for i in range(batches):
            b = i % 2
            with torch.npu.stream(st):
                bufs[b].copy_(hs[b], non_blocking=True)
            ev_h[b].record(st)
            sc.wait_event(ev_h[b])
            with torch.npu.stream(sc):
                out = (bufs[b] @ bufs[b]).sum()
            ev_c[b].record(sc)
            sd.wait_event(ev_c[b])
            with torch.npu.stream(sd):
                out_cpu = out.cpu()
        torch.npu.synchronize()
        pipe = _t() - t0
        t1 = _t()
        for i in range(batches):
            dd = hs[i % 2].to("npu")
            (dd @ dd).sum().cpu()
        torch.npu.synchronize()
        serial = _t() - t1
        return pipe, serial, (serial - pipe) / serial if serial > 0 else 0

    p9, s9, o9 = run_pipeline_n(4096, 10)
    res["timing"]["s9_pipe_4096x10_s"] = round(p9, 6)
    res["timing"]["s9_serial_4096x10_s"] = round(s9, 6)
    res["timing"]["s9_overlap_4096"] = round(o9, 4)
    print(f"[S9] 4096²×10批(预热后): pipe={p9*1e3:.2f}ms serial={s9*1e3:.2f}ms 重叠率={o9:.1%}")

    # ── 结论判定 ──
    o7 = res["timing"]["s7_overlap_512"]
    o8 = res["timing"]["s8_overlap_2048"]
    if async_ratio >= 0.5:
        conclusion = "退化为同步：copy_ non_blocking 入队即等待（torch_npu H2D 异步缺口）"
    elif res["timing"]["s6_per_event_ms"] > 1.0:
        conclusion = f"事件链开销主导：单次 record+wait ≈ {res['timing']['s6_per_event_ms']}ms"
    elif o9 > 0.05:
        conclusion = (f"同步开销主导（已证实）：事件链固定开销 {res['timing']['s6_per_event_ms']}ms/次 vs "
                      f"小负载单阶段 ~0.1ms → 小粒度(512²={o7:.1%}/2048²={o8:.1%})被掩盖；"
                      f"大负载 4096²×10 重叠率 {o9:.1%} → 真实推理负载粒度下流水线有效")
    elif o8 > 0.05 and o7 <= 0:
        conclusion = f"同步开销主导：小粒度被掩盖，大张量(2048²)重叠率 {o8:.1%} → 需增大计算粒度"
    else:
        conclusion = f"未明确定位：512²={o7:.1%} 2048²={o8:.1%} 4096²={o9:.1%}，需进一步细化"
    res["note"] = conclusion
    res["verdict"] = "BREAKDOWN_DONE"
    print(f"\n结论: {conclusion}")

    with open("double_buffer_breakdown_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入 double_buffer_breakdown_result.json")


if __name__ == "__main__":
    main()
