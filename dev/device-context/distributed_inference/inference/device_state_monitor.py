#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""device_state_monitor.py — D11 集成：推理服务设备状态监控（独立进程，无侵入）

【做什么】serve 运行期间旁路监控设备活性与四态：
  · 周期 probe_device（轻量活性探针）+ query_device_state（四态查询）
  · 状态变化（AVAILABLE→DEGRADED→ISOLATED）时打印事件
  · 全程落 JSON（供后续分析/看板）

【验证点】serve 运行中监控进程持续输出 probe_ok=True / state=available，
         人为注入压力/故障时观察到状态转换事件。

【用法】serve 启动后并行：
    python3 device_state_monitor.py --ordinal 0 --interval 5 --out device_state_monitor.json

【效果】A10 从"模块级验证"升级为"serve 长驻期间持续可观测" —— 运维可见、可对接告警。
"""
import argparse
import json
import os
import sys
import time


def _conformance_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(here, "ascend_regression", "conformance"),
        os.path.join(here, "..", "ascend_regression", "conformance"),
        "/mnt/raid/hliu553/runtime-team/dev/device-context/benchmarks/ascend_regression/conformance",
    ):
        if os.path.exists(os.path.join(cand, "device_state.py")):
            return cand
    return here


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0=无限")
    ap.add_argument("--out", default="device_state_monitor.json")
    args = ap.parse_args()

    sys.path.insert(0, _conformance_dir())
    import torch  # noqa: E402
    import torch_npu  # noqa: F401,E402
    from device_state import query_device_state  # noqa: E402
    from recovery import probe_device  # noqa: E402

    def sync_fn():
        torch.npu.synchronize()

    history, last_state, t0 = [], None, time.time()
    print(f"[monitor] 开始监控 device:{args.ordinal} interval={args.interval}s", flush=True)

    while args.max_seconds == 0 or time.time() - t0 < args.max_seconds:
        try:
            ok = probe_device(args.ordinal, device="npu", sync_fn=sync_fn)
            st = query_device_state(args.ordinal)
        except Exception as e:  # noqa: BLE001
            print(f"[monitor][WARN] 探测失败: {e}", flush=True)
            ok, st = False, None
            time.sleep(args.interval)
            continue

        rec = {"t_s": round(time.time() - t0, 1), "probe_ok": ok,
               "state": st.value if st else "unknown"}
        history.append(rec)

        if st != last_state:
            print(f"[monitor][EVENT] state={st.value if st else '?'} "
                  f"probe_ok={ok}（自 {last_state} 变化）", flush=True)
            last_state = st
        time.sleep(args.interval)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ordinal": args.ordinal, "interval": args.interval,
                   "samples": len(history), "history": history},
                  f, ensure_ascii=False, indent=2)
    print(f"[monitor] 结束，{len(history)} 个采样点 → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
