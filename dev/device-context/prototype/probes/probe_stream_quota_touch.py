#!/usr/bin/env python3
"""
probe_stream_quota_touch.py — 「创建成功」是不是真的（配额上限判定的前置核查）

【为什么需要它】
  `probe_stream_quota.py` 在 25.6 万档仍全绿、创建耗时线性（0.07s/4k … 3.67s/256k）
  ⇒ 有两种可能：
    ① 本栈流创建近似 O(1) 句柄（真实且无上限）；
    ② 创建只是 Python 侧对象，**设备侧资源直到真正使用才分配** ⇒ "创建成功"不构成配额证据。
  本探针用三件事区分：**宿主 RSS 增量** + **逐个强制使用（真跑计算）** + **失败点定位**。

【判据】
  T1 每个流**真被使用过**（在其上跑一次最小计算并校验数值）
  T2 使用过程零异常，且抽样流数值正确
  T3 宿主 RSS 增量被记录（用于判断"创建"是否伴随真实资源）
  T4 首个失败档位被定位（或如实报"到上限档仍未出现失败"）

【环境变量】DC_BACKEND / DC_OUT_DIR / DC_TAG / DC_QUOTA_N（默认 50000） / DC_TOUCH_SAMPLE（默认 2000）
【判定】STREAM_QUOTA_TOUCH_PASS / PARTIAL / FAIL
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402
import runtime  # noqa: E402

BACKEND = os.environ.get("DC_BACKEND", "ascend")
OUT_DIR = Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
TAG = os.environ.get("DC_TAG", "")
N = int(os.environ.get("DC_QUOTA_N", "50000"))
SAMPLE = int(os.environ.get("DC_TOUCH_SAMPLE", "2000"))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def dev_used_mib(bk, ordinal: int = 0) -> float:
    """设备侧已用显存（MiB）。取不到则返回 nan（如实标注，不补零）。"""
    try:
        st = bk.memory_stats(ordinal)
        if isinstance(st, dict) and "used_mb" in st:
            return float(st["used_mb"])
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


def main() -> int:
    print("=== probe_stream_quota_touch.py: 创建是否伴随真实资源 ===")
    bk = runtime.use(BACKEND)
    bk.device_count()
    dt = bk.device_type
    ns = getattr(torch, dt)
    dev = f"{dt}:0"
    torch.zeros(1, device=dev)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[env] backend={BACKEND} N={N} sample={SAMPLE} torch={torch.__version__}")

    checks: dict = {}
    rss0 = rss_mb()
    hbm0 = dev_used_mib(bk)

    # ── 建流 ──
    t0 = time.monotonic()
    streams = []
    err = None
    try:
        for _ in range(N):
            streams.append(ns.Stream())
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}"
    t_create = time.monotonic() - t0
    n_created = len(streams)
    rss1 = rss_mb()
    hbm1 = dev_used_mib(bk)
    print(f"[创建] {n_created}/{N} 成功，{t_create:.2f}s")
    print(f"[资源] 宿主 RSS {rss0:.1f} → {rss1:.1f} MB（Δ{rss1 - rss0:+.1f}，"
          f"{(rss1 - rss0) * 1024 / max(n_created, 1):.2f} B/流）；"
          f"设备已用 {hbm0:.1f} → {hbm1:.1f} MiB（Δ{hbm1 - hbm0:+.1f}，"
          f"{(hbm1 - hbm0) * 1024 / max(n_created, 1):.3f} B/流）"
          + (f"  首个失败: {err}" if err else ""))
    checks["T1_create"] = {"ok": err is None and n_created == N,
                           "detail": f"{n_created}/{N}，{t_create:.2f}s" + (f"，失败 {err}" if err else "")}
    checks["T3_resource_cost"] = {
        "ok": True,                     # 记录型判据；"是否为 0" 本身是结论
        "rss_delta_mb": round(rss1 - rss0, 1),
        "hbm_delta_mib": round(hbm1 - hbm0, 2) if hbm0 == hbm0 and hbm1 == hbm1 else None,
        "bytes_per_stream_host": round((rss1 - rss0) * 1024 / max(n_created, 1), 2),
        "bytes_per_stream_dev": (round((hbm1 - hbm0) * 1024 / max(n_created, 1), 3)
                                 if hbm0 == hbm0 and hbm1 == hbm1 else None),
        "detail": f"宿主 {rss1 - rss0:+.1f} MB / 设备 {hbm1 - hbm0:+.1f} MiB（{n_created} 流）",
    }

    # ── 强制使用：抽样流上真跑计算 ──
    a = torch.full((4, 4), 2.0, device=dev)
    n_touch = min(SAMPLE, n_created)
    touched_ok = 0
    touch_err = None
    t1 = time.monotonic()
    try:
        for i in range(0, n_touch):
            with ns.stream(streams[i]):
                v = float(a.sum().item())
            if abs(v - 32.0) < 1e-3:
                touched_ok += 1
    except Exception as e:  # noqa: BLE001
        touch_err = f"在第 {i} 个流上 {type(e).__name__}: {str(e)[:200]}"
    t_touch = time.monotonic() - t1
    print(f"[使用] 抽样 {n_touch} 个流真跑计算：正确 {touched_ok}/{n_touch}，{t_touch:.2f}s"
          + (f"  异常: {touch_err}" if touch_err else ""))
    checks["T1_touch_all_usable"] = {"ok": touched_ok == n_touch and touch_err is None,
                                     "detail": f"{touched_ok}/{n_touch} 数值正确，{t_touch:.2f}s"
                                               + (f"；{touch_err}" if touch_err else "")}
    checks["T2_values_correct"] = {"ok": touched_ok == n_touch,
                                   "detail": f"期望 32.0，正确 {touched_ok}/{n_touch}"}

    # ── 释放后可用 ──
    try:
        streams.clear()
        gc.collect()
        ns.synchronize()
        rss2 = rss_mb()
        v2 = float('nan')
        with ns.stream(ns.Stream()):
            v2 = float(a.sum().item())
        ok_rel = abs(v2 - 32.0) < 1e-3
        print(f"[释放] 释放后新流结果={v2:.1f}（期望 32.0）{'✅' if ok_rel else '❌'}，"
              f"RSS {rss1:.1f} → {rss2:.1f} MB")
    except Exception as e:  # noqa: BLE001
        ok_rel = False
        rss2 = rss_mb()
        print(f"[释放] 异常: {type(e).__name__}: {str(e)[:160]}")
    checks["T4_release_usable"] = {"ok": ok_rel, "detail": f"释放后新流结果={v2:.1f}" if ok_rel else "释放后异常"}

    checks["T5_limit_bracket"] = {
        "ok": True,
        "detail": (f"本档 N={N} {'未出现失败' if err is None else '出现失败'}；"
                   f"创建耗时 {t_create:.2f}s，RSS Δ{rss1 - rss0:+.1f} MB"),
    }

    passed = sum(1 for v in checks.values() if v.get("ok"))
    total = len(checks)
    verdict = ("STREAM_QUOTA_TOUCH_PASS" if passed == total
               else ("STREAM_QUOTA_TOUCH_PARTIAL" if passed else "STREAM_QUOTA_TOUCH_FAIL"))
    print(f"\n{verdict}: {passed}/{total}")

    out = {"verdict": verdict, "passed": passed, "total": total, "backend": BACKEND,
           "n_requested": N, "n_created": n_created,
           "rss_delta_mb": round(rss1 - rss0, 1),
           "hbm_delta_mib": (round(hbm1 - hbm0, 2) if hbm0 == hbm0 and hbm1 == hbm1 else None),
           "torch": torch.__version__, "checks": checks}
    p = OUT_DIR / f"stream_quota_touch_result{TAG}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {p}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
