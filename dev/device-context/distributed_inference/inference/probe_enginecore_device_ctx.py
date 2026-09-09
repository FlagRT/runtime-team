#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_enginecore_device_ctx.py — P3 · A8 验收：EngineCore 子进程设备句柄/上下文

【职责对应】D2 统一设备句柄 + 生命周期管理
【验收标准】A8：EngineCore 子进程设备句柄/上下文可用
【取证思路】坑 A2 告诉我们：EngineCore 是 spawn 子进程，主进程读不到它的 stats。
           所以不依赖 vLLM 内部接口，改从 /proc/<pid> 直接取证设备句柄：
             1. /proc/<pid>/fd   → 是否持有 /dev/davinci* 设备句柄（fd 级证据）
             2. /proc/<pid>/maps → 是否映射设备内存（映射级证据）
             3. /proc/<pid>/environ → 设备枚举相关环境变量（ASCEND_VISIBLE_DEVICES 等）
             4. 主进程 vs 子进程对照 → 验证"设备上下文归属子进程"
             5. 功能级证据：发 1 条请求成功 = 子进程设备上下文真正可用
【用法】容器内：python3 probe_enginecore_device_ctx.py [--port 8100] [--out ...]
【判定】ENGINECORE_CTX_PASS = 子进程存在 + 持有 NPU 设备 fd + 功能请求成功
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request

DEV_KEYWORDS = ("davinci", "npu", "devmm", "ascend", "hdcdrv")
ENV_KEYS = (
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "DEVICE_ID",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "TP_SIZE",
    "VLLM_USE_V1",
    "NPU_VISIBLE_DEVICES",
)


def list_procs():
    out = subprocess.run(
        ["ps", "-eo", "pid,ppid,cmd"], capture_output=True, text=True
    ).stdout
    procs = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        procs.append(
            {"pid": int(parts[0]), "ppid": int(parts[1]), "cmd": parts[2]}
        )
    return procs


def find_engine_core(procs):
    hits = []
    for p in procs:
        if "VLLM::EngineCore" in p["cmd"] and "grep" not in p["cmd"]:
            hits.append(p)
    return hits


def find_serve_main(procs):
    for p in procs:
        if "vllm serve" in p["cmd"] and "grep" not in p["cmd"]:
            return p
    return None


def device_fds(pid):
    d = f"/proc/{pid}/fd"
    fds = []
    err = None
    try:
        for fd in os.listdir(d):
            try:
                tgt = os.readlink(os.path.join(d, fd))
            except OSError:
                continue
            low = tgt.lower()
            if any(k in low for k in DEV_KEYWORDS):
                fds.append(tgt)
    except OSError as e:
        err = str(e)
    return sorted(set(fds)), err


def device_maps(pid):
    path = f"/proc/{pid}/maps"
    hits = set()
    err = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                low = line.lower()
                if any(k in low for k in DEV_KEYWORDS):
                    parts = line.split()
                    hits.add(parts[-1] if len(parts) > 5 else line.strip()[:60])
    except OSError as e:
        err = str(e)
    return sorted(hits), err


def env_of(pid):
    path = f"/proc/{pid}/environ"
    env = {}
    try:
        with open(path, "rb") as f:
            for item in f.read().split(b"\x00"):
                if b"=" in item:
                    k, v = item.split(b"=", 1)
                    env[k.decode(errors="replace")] = v.decode(errors="replace")
    except OSError:
        return {}
    return {k: env.get(k) for k in ENV_KEYS if k in env}


def rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None
    return None


def func_probe(port):
    """功能级证据：子进程设备上下文可用 ⇒ 请求能正常完成"""
    payload = json.dumps(
        {
            "model": "qwen3-4b",
            "prompt": "The capital of France is",
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        r = json.loads(resp.read().decode("utf-8"))
    return {
        "ok": True,
        "text": r["choices"][0]["text"][:120],
        "latency_s": round(time.time() - t0, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--out", default="enginecore_device_ctx_result.json")
    args = ap.parse_args()

    result = {
        "verdict": "ENGINECORE_CTX_FAIL",
        "checks": {},
        "engine_core": [],
        "main_process": {},
        "functional": {},
        "note": "",
    }

    procs = list_procs()
    ecs = find_engine_core(procs)
    main = find_serve_main(procs)

    print(f"[1/4] 发现 EngineCore 子进程 {len(ecs)} 个")
    for ec in ecs:
        fds, fd_err = device_fds(ec["pid"])
        maps, map_err = device_maps(ec["pid"])
        info = {
            "pid": ec["pid"],
            "ppid": ec["ppid"],
            "cmd": ec["cmd"][:160],
            "device_fds": fds,
            "device_fd_count": len(fds),
            "device_maps": maps,
            "device_map_count": len(maps),
            "env": env_of(ec["pid"]),
            "rss_mb": rss_mb(ec["pid"]),
        }
        if fd_err:
            info["fd_error"] = fd_err
        if map_err:
            info["maps_error"] = map_err
        result["engine_core"].append(info)
        print(
            f"   pid={ec['pid']} ppid={ec['ppid']} "
            f"设备fd={len(fds)} 设备映射={len(maps)} RSS={info['rss_mb']}MB"
        )
        for t in fds[:5]:
            print(f"      fd → {t}")

    print("[2/4] 主进程对照（验证设备上下文归属子进程）")
    if main:
        fds, _ = device_fds(main["pid"])
        maps, _ = device_maps(main["pid"])
        result["main_process"] = {
            "pid": main["pid"],
            "cmd": main["cmd"][:160],
            "device_fds": fds,
            "device_fd_count": len(fds),
            "device_map_count": len(maps),
            "env": env_of(main["pid"]),
        }
        print(f"   main pid={main['pid']} 设备fd={len(fds)}")
    else:
        result["note"] += "未找到 vllm serve 主进程；"
        print("   未找到主进程")

    print("[3/4] 功能级证据：发 1 条请求（子进程设备上下文可用则应成功）")
    try:
        result["functional"] = func_probe(args.port)
        print(f"   {result['functional']}")
    except Exception as e:  # noqa: BLE001
        result["functional"] = {"ok": False, "error": str(e)}
        print(f"   请求失败: {e}")

    print("[4/4] 判定")
    has_ec = len(ecs) > 0
    ec_holds_dev = any(e["device_fd_count"] > 0 for e in result["engine_core"])
    func_ok = result["functional"].get("ok", False)
    result["checks"] = {
        "engine_core_spawned": has_ec,
        "engine_core_holds_npu_fd": ec_holds_dev,
        "functional_request_ok": func_ok,
    }
    ok = has_ec and ec_holds_dev and func_ok
    result["verdict"] = "ENGINECORE_CTX_PASS" if ok else "ENGINECORE_CTX_FAIL"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 判定：{result['verdict']} ===")
    print(f"子进程 spawn: {has_ec} | 持有 NPU 设备 fd: {ec_holds_dev} | 功能请求: {func_ok}")
    print(f"结果：{args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
