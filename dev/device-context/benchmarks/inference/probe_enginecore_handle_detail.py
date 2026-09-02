#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_enginecore_handle_detail.py — A8 补验：句柄/映射的具体含义与释放行为

【原验证的不足】原 `probe_enginecore_device_ctx.py` 只给出"fd=1、映射=124 处"的计数，
  · 未说明这 124 处映射**到底是什么**（哪些设备文件、哪些内存区）
  · 未验证**进程退出后句柄是否释放**（资源泄漏是真实运维风险）

【本脚本】
  1. fd 明细：逐个列出 /proc/<pid>/fd 指向，按类型归类（davinci / devmm / 其他）
  2. maps 明细：按映射路径归类统计，显示 Top 设备文件路径
  3. 线程/子进程：列出 EngineCore 派生的 worker 进程（TP>1 时每个 rank 一个）
  4. 释放检查：--after-stop 模式下检查目标 pid 是否消失、残留进程数

【用法】
  # 服务运行中采集
  python3 probe_enginecore_handle_detail.py --out enginecore_detail_running.json
  # 服务停止后采集（验证释放）
  python3 probe_enginecore_handle_detail.py --after-stop --out enginecore_detail_stopped.json
"""
import argparse
import json
import os
import re
import subprocess
from collections import Counter


def find_procs(pattern):
    out = subprocess.run(["ps", "-eo", "pid,ppid,cmd"],
                         capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        if pattern in parts[2] and "grep" not in parts[2]:
            hits.append({"pid": int(parts[0]), "ppid": int(parts[1]),
                         "cmd": parts[2][:160]})
    return hits


def fd_detail(pid):
    d = f"/proc/{pid}/fd"
    items = []
    try:
        for fd in os.listdir(d):
            try:
                tgt = os.readlink(os.path.join(d, fd))
            except OSError:
                continue
            items.append(tgt)
    except OSError:
        return {"error": "no access"}
    cats = Counter()
    for t in items:
        low = t.lower()
        if "davinci" in low:
            cats["davinci"] += 1
        elif "devmm" in low:
            cats["devmm"] += 1
        elif "npu" in low or "ascend" in low:
            cats["npu_ascend"] += 1
        elif low.startswith("/"):
            cats["file"] += 1
        else:
            cats["other"] += 1
    davinci = sorted({t for t in items if "davinci" in t.lower()})
    return {"total": len(items), "by_type": dict(cats),
            "davinci_paths": davinci[:12]}


def maps_detail(pid):
    path = f"/proc/{pid}/maps"
    cnt = Counter()
    samples = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                p = parts[-1]
                low = p.lower()
                if "davinci" in low:
                    key = "davinci"
                elif "devmm" in low:
                    key = "devmm"
                elif "npu" in low or "ascend" in low:
                    key = "npu_ascend"
                elif p.startswith("["):
                    key = "kernel_special"
                elif p.startswith("/"):
                    key = "file"
                else:
                    key = "anon"
                cnt[key] += 1
                samples.setdefault(key, p)
    except OSError:
        return {"error": "no access"}
    return {"total": sum(cnt.values()), "by_type": dict(cnt),
            "sample_path_per_type": {k: v[:120] for k, v in samples.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="enginecore_handle_detail.json")
    ap.add_argument("--after-stop", action="store_true")
    args = ap.parse_args()

    cores = find_procs("VLLM::EngineCore")
    workers = find_procs("Worker_TP")
    result = {
        "mode": "after_stop" if args.after_stop else "running",
        "engine_core_procs": cores,
        "worker_procs": workers,
        "detail": [],
    }

    for p in cores:
        result["detail"].append({
            "pid": p["pid"],
            "fd": fd_detail(p["pid"]),
            "maps": maps_detail(p["pid"]),
        })
    for p in workers[:4]:
        result["detail"].append({
            "pid": p["pid"], "role": "worker",
            "cmd": p["cmd"][:100],
            "fd": fd_detail(p["pid"]),
            "maps": None,
        })

    if args.after_stop:
        result["release_check"] = {
            "engine_core_remaining": len(cores),
            "worker_remaining": len(workers),
            "released": len(cores) == 0 and len(workers) == 0,
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"=== A8 句柄明细（{result['mode']}）===")
    print(f"EngineCore 进程: {len(cores)}  Worker 进程: {len(workers)}")
    for d in result["detail"]:
        tag = d.get("role", "engine_core")
        print(f"  [{tag}] pid={d['pid']}")
        print(f"      fd: {d['fd']}")
        if d.get("maps"):
            print(f"      maps: {d['maps']}")
    if args.after_stop:
        print(f"释放检查: {result['release_check']}")
    print(f"结果：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
