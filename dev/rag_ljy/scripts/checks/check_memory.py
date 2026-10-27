#!/usr/bin/env python3
"""Read accelerator memory through device-context and save one compact report."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
import selectors
import subprocess
import time
from uuid import uuid4

from _bootstrap import load_project_env, PROJECT_ROOT
from rag_engine.memory_probe import DeviceMemoryProbe


def check_cross_process(probe, allocation_mib):
    """Hold a small allocation in a child; observe it through runtime in parent."""
    before = probe.sample()
    if before["status"] != "ok" or before["stats"]["free_mb"] < allocation_mib + 512:
        raise RuntimeError("Need a valid baseline and at least allocation + 512 MiB free")
    code = """
import sys
from rag_engine.device import prepare_torch_device
torch, device = prepare_torch_device(sys.argv[1])
import runtime
tensor = torch.empty(int(sys.argv[2]) * 1024 * 1024, dtype=torch.uint8, device=device)
runtime.synchronize()
print('MEMORY_PROBE_ALLOCATION_READY', flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-u", "-c", code, probe.device, str(allocation_mib)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    messages = []
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + 40
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise RuntimeError("Timed out waiting for allocation child")
                line = child.stdout.readline().strip()
                messages.append(line)
                if line == "MEMORY_PROBE_ALLOCATION_READY":
                    break
                if not line and child.poll() is not None:
                    raise RuntimeError("Allocation child failed: " + " / ".join(messages[-5:]))
        held = probe.sample()
    finally:
        try:
            if child.poll() is None:
                child.stdin.close()
                child.wait(timeout=15)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()
            child.stdout.close()
    after = probe.sample()
    if held["status"] != "ok" or after["status"] != "ok":
        raise RuntimeError("Memory read failed during cross-process test")
    deltas = {
        "used_increase_mib": held["stats"]["used_mb"] - before["stats"]["used_mb"],
        "free_decrease_mib": before["stats"]["free_mb"] - held["stats"]["free_mb"],
        "used_release_mib": held["stats"]["used_mb"] - after["stats"]["used_mb"],
        "free_recovery_mib": after["stats"]["free_mb"] - held["stats"]["free_mb"],
    }
    passed = child.returncode == 0 and all(v >= allocation_mib - 2 for v in deltas.values())
    return {"status": "passed" if passed else "failed", "allocation_mib": allocation_mib,
            "child_pid": child.pid, "child_exit_code": child.returncode,
            "before": before, "held": held, "after": after, "deltas": deltas,
            "note": "Differences include the child runtime/context allocation; scope is validated only for this backend/run, not for fallback paths."}


def main():
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.getenv("RAG_DEVICE", "auto"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--cross-process-mib", type=int, default=0,
                        help="verify visibility of a separate process holding 1..256 MiB (default off)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts/checks/memory_probe")
    args = parser.parse_args()
    if not 1 <= args.count <= 1000 or not math.isfinite(args.interval) or args.interval <= 0 or not 0 <= args.cross_process_mib <= 256:
        parser.error("count must be 1..1000, interval finite/positive, cross-process-mib 0..256")
    probe = DeviceMemoryProbe(args.device)
    samples = []
    for index in range(args.count):
        if index:
            time.sleep(args.interval)
        sample = probe.sample()
        samples.append(sample)
        print(json.dumps(sample, ensure_ascii=False), flush=True)
    now = datetime.now(timezone(timedelta(hours=8)))
    output = args.output_dir / f"{now:%Y-%m-%d_%H%M%S}-{uuid4().hex[:6]}"
    output.mkdir(parents=True)
    cross_process = None
    if args.cross_process_mib:
        try:
            cross_process = check_cross_process(probe, args.cross_process_mib)
        except Exception as error:
            cross_process = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        print("cross_process: " + json.dumps(cross_process, ensure_ascii=False), flush=True)
    passed = all(s["status"] == "ok" for s in samples) and (cross_process is None or cross_process["status"] == "passed")
    lines = ["# Device-context 设备内存探测", "",
             f"日期：{now.isoformat()}（北京时间）；探测结果：{'passed' if passed else 'failed'}。", "",
             f"请求设备：`{args.device}`；采样 {args.count} 次，间隔 {args.interval}s。",
             "调用路径：RAG → device-context runtime.memory_stats → 所选平台后端。", "",
             "| 时间（UTC） | 状态 | 设备 | 总量（MiB） | 已用（MiB） | 可用（MiB） | 采样耗时（ms） |",
             "|---|---|---|---|---|---|---|"]
    for sample in samples:
        stats = sample.get("stats") or {}
        lines.append(f"| {sample['sampled_at']} | {sample['status']} | {sample['device']} | "
                     f"{stats.get('total_mb', '未知')} | {stats.get('used_mb', '未知')} | "
                     f"{stats.get('free_mb', '未知')} | {sample['sample_duration_ms']} |")
    if cross_process is not None:
        lines.extend(["", "## 独立进程内存占用验证", "",
                      "子进程通过device-context绑定同一逻辑设备，分配并持有小块内存；父进程通过统一内存接口观测分配与释放。",
                      "变化量包含子进程运行环境的额外占用。此次验证不证明所有版本、平台及兼容回退都能报告整卡内存。", "",
                      "```json", json.dumps(cross_process, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "统计范围标记为 backend_reported_unverified：当前统一接口未提供底层查询/回退路径的元数据。",
                  "Ascend 后端查询失败可能回退到当前进程分配量；这些读数尚未用于内存准入控制。",
                  "该接口测量内存占用，不提供加速卡计算利用率。", "", "## 原始采样（仅小型指标记录）", "", "```json",
                  json.dumps(samples, ensure_ascii=False, indent=2), "```", ""])
    (output / "report.md").write_text("\n".join(lines))
    print(f"Report: {output / 'report.md'}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
