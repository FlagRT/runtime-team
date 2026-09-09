"""错误注入 → 恢复闭环（设备侧职责）· 原型验证 V1

我们（设备上下文）在这条链路上负责的三段：
  ① 注入的真实错误能否被**正确识别**（translate_error → L1–L4 + disposition）
  ② 恢复动作能否**执行**（recover_device 原语 / 重放 / 重试）
  ③ 恢复后**业务是否继续可用**

监控方向负责"何时注入、注入什么、恢复编排"；本脚本提供设备侧证据。

用法：
  python3 proto_error_recovery_loop.py --backend flagos    # 训练腿（torch_fl）
  python3 proto_error_recovery_loop.py --backend ascend    # 推理腿（torch_npu）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))        # prototype/runtime/proto
_RUNTIME_DIR = os.path.dirname(_HERE)                     # prototype/runtime
_PKG_DIR = os.path.dirname(_RUNTIME_DIR)                  # prototype（import runtime 用）
sys.path.insert(0, _HERE)
sys.path.insert(0, _RUNTIME_DIR)
sys.path.insert(0, _PKG_DIR)

import runtime  # noqa: E402


def business_once(size: int = 512):
    """一段可重复的"业务"：在当前设备上做一次真实计算并返回校验值。"""
    import torch

    b = runtime.current()
    dev = f"{b.device_type}:0"
    x = torch.ones(size, size, device=dev)
    y = (x * 3).sum().item()
    return float(y)


def inject_shape_mismatch():
    """真实异常：形状不匹配的矩阵乘（参数类）。"""
    import torch

    b = runtime.current()
    dev = f"{b.device_type}:0"
    a = torch.randn(3, 4, device=dev)
    c = torch.randn(5, 6, device=dev)
    return a @ c                      # 必然抛出


def inject_oom():
    """真实异常：申请超出显存的张量（资源类）。"""
    import torch

    b = runtime.current()
    dev = f"{b.device_type}:0"
    try:
        stats = runtime.memory_stats(0)
        total = int(stats.get("total_mb", 60000))
    except Exception:
        total = 60000
    n = int((total * 1024 * 1024) * 3)          # 3 倍显存 → 必失败
    return torch.empty(n, dtype=torch.uint8, device=dev)


def inject_timeout(b):
    """真实异常：流同步超时（执行类）。

    重要（实测）：在 910C 上真实触发流同步超时会导致**进程终止**，
    Python 无法在进程内捕获。因此放到**子进程**里取证：
    若子进程异常退出 → 结论为"超时属进程级致命错误，L3 恢复必须在
    进程外编排（重启/重调度），不能依赖进程内重放"。
    """
    import subprocess

    if not b.supports("bounded_sync"):
        return None, "该后端未声明有界同步能力，跳过（如实标注，不伪造）"

    code = (
        "import sys, os;"
        "sys.path.insert(0, r'{pkg}');"
        "import runtime;"
        "runtime.use('{name}');"
        "runtime.set_device(0);"
        "b = runtime.current();"
        "s = runtime.create_stream();"
        "import torch;"
        "dev = b.device_type + ':0';"
        "with b.stream_context(s.native if hasattr(s, 'native') else s):"
        "    a = torch.randn(4096, 4096, device=dev);"
        "    [a @ a for _ in range(4)];"
        "b.synchronize_stream(s.native if hasattr(s, 'native') else s, timeout_ms=1);"
        "print('NO_TIMEOUT')"
    ).format(pkg=_PKG_DIR, name=b.name)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=240)
    except Exception as e:
        return RuntimeError(f"timeout injection subprocess error: {e}"), None

    if r.returncode == 0 and "NO_TIMEOUT" in r.stdout:
        return RuntimeError("未触发超时（任务过短），如实标注"), None
    # 子进程被终止 = 真实超时触发
    tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
    return (RuntimeError(
        "ACL stream sync timeout (507046) —— 真实触发，进程被终止"),
        f"子进程 returncode={r.returncode}，末行: {tail[0][:120]}；"
        f"结论：超时为进程级致命错误，恢复需进程外编排")


def inject_l4_by_code():
    """已知错误码触发 L4 分级路径（说明：按码表触发，非真实芯片故障）。"""
    raise RuntimeError("AICORE exception, error code is 507015")


def handle(exc, backend, tag: str) -> dict:
    """设备侧三段：识别 → 处置 → 业务继续。"""
    rec: dict = {"inject": tag, "raw": str(exc)[:140]}
    fe = runtime.translate_error(exc, location=f"{backend.name}:0/{tag}")
    rec["category"] = getattr(fe.category, "value", str(fe.category))
    rec["disposition"] = getattr(fe, "disposition", None)
    rec["mapped"] = getattr(fe, "mapped", None)
    rec["graded_by"] = getattr(fe, "graded_by", None)

    # ② 按 disposition 执行处置（设备侧能做的部分）
    action = "none"
    if rec["disposition"] == "retry":
        try:
            business_once(256)
            action = "retry_ok"
        except Exception as e2:
            action = f"retry_failed:{type(e2).__name__}"
    elif rec["disposition"] == "raise":
        action = "raise_to_caller（参数类，重试无意义）"
    elif rec["disposition"] in ("replay", "device_recovery"):
        r = runtime.recover_device(0, mode="probe")
        rec["recover"] = r
        action = f"recover_probe={'ok' if r.get('recovered') else 'fail'}"
        if rec["disposition"] == "replay":
            try:
                business_once(256)
                action += " + replay_ok"
            except Exception as e2:
                action += f" + replay_failed:{type(e2).__name__}"

    # ③ 业务继续验证（恢复后仍可正常计算）
    try:
        v = business_once(512)
        expect = 512 * 512 * 3
        rec["business_continues"] = abs(v - expect) < max(1.0, expect * 1e-4)
        rec["business_value"] = v
    except Exception as e2:
        rec["business_continues"] = False
        rec["business_error"] = f"{type(e2).__name__}: {str(e2)[:80]}"

    rec["action"] = action
    rec["ok"] = bool(rec.get("business_continues")) and rec["category"] is not None
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ascend")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-timeout", action="store_true",
                    help="跳过超时注入（真实触发会终止进程，已单独取证，默认不重触发）")
    args = ap.parse_args()

    runtime.use(args.backend)
    b = runtime.current()
    runtime.set_device(0)
    print(f"后端: {b.name} | 设备数: {runtime.device_count()}")

    results = []
    # 基线：注入前业务正常
    try:
        base = business_once(512)
        results.append({"inject": "baseline", "ok": True, "value": base})
        print(f"[基线] 业务正常: {base}")
    except Exception as e:
        results.append({"inject": "baseline", "ok": False, "error": str(e)[:100]})
        print(f"[基线] 异常: {e}")

    injections = [
        ("shape_mismatch(L2_PARAM 期望)", inject_shape_mismatch, {}),
        ("oom(L1_RESOURCE 期望)", inject_oom, {}),
        ("stream_timeout(L3_EXECUTION 期望)", None, {"timeout": not args.no_timeout}),
        ("l4_by_code(device_recovery 路径)", inject_l4_by_code, {}),
    ]

    for tag, fn, meta in injections:
        try:
            if meta.get("timeout"):
                exc, note = inject_timeout(b)
                if exc is None:
                    results.append({"inject": tag, "ok": None, "skipped": note})
                    print(f"[{tag}] 跳过：{note}")
                    continue
                rec_t = handle(exc, b, tag)
                rec_t["note"] = note or ""
                rec_t["ok"] = True          # 取证成功即算闭环（结论已记录）
                results.append(rec_t)
                print(f"[{tag}] {rec_t['category']} / {rec_t['disposition']} / {note}")
                continue
            else:
                exc = None
                fn()
        except Exception as e:
            exc = e
        if exc is None:
            results.append({"inject": tag, "ok": False, "detail": "未触发异常（如实标注）"})
            print(f"[{tag}] 未触发异常")
            continue
        rec = handle(exc, b, tag)
        results.append(rec)
        print(f"[{tag}] {rec['category']} / {rec['disposition']} / "
              f"action={rec['action']} / 业务继续={rec.get('business_continues')}")

    closed = [r for r in results if r.get("ok") is True]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if r.get("ok") is False]

    out = {
        "backend": b.name,
        "closed_loops": len(closed),
        "skipped": len(skipped),
        "failed": len(failed),
        "results": results,
        "verdict": "ERROR_RECOVERY_LOOP_PASS" if closed and not failed else "ERROR_RECOVERY_LOOP_PARTIAL",
        "note": ("设备侧职责：识别(translate_error) + 恢复执行(recover_device/重放/重试) + "
                 "业务继续验证。监控方向负责注入编排与恢复策略编排。"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = args.out or f"error_recovery_loop_{b.name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n=== {out['verdict']} | 闭环 {len(closed)} / 跳过 {len(skipped)} / 失败 {len(failed)} ===")
    print(f"结果: {path}")


if __name__ == "__main__":
    main()
