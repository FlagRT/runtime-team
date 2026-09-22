#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_stream_quota.py — S-16「流数量配额」验证（**后端无关**）
═══════════════════════════════════════════════════════════════════════════════

【为什么单独一个探针】
  多流 Stream 验收基线（S-1 ~ S-16）里，S-16 的判据是"**流数量配额**：连续创建大量流
  不报错、且已创建的流仍可用、释放后无残留"。它既不在 `probe_stream_semantics_full.py`
  覆盖范围（那 8 项是 S1/S2 补强 + S8~S13），也不在 conformance 用例里
  （S-10 查的是"创建销毁循环后新流仍可用"，不查"同时在世的数量上限"）。
  ⇒ 独立成资产，供各实例用**同一口径**取数（910C / P800 已按 2000 流口径测过）。

【判据】
  Q1 连续创建 N 个流全部成功（N 默认 2000，与 910C / P800 口径一致）
  Q2 第 1 个流仍可正常执行（在其上做一次真实计算并校验数值）
  Q3 全部释放后无异常，且能再创建新流并正常执行（无配额泄漏）

【环境变量】
  DC_BACKEND / DC_DEV_API / DC_OUT_DIR / DC_TAG   同其它探针
  DC_QUOTA_N   流的个数（默认 2000）

【用法】
  DC_BACKEND=cambricon MLU_VISIBLE_DEVICES=0 python3 probe_stream_quota.py

【判定】STREAM_QUOTA_PASS / STREAM_QUOTA_PARTIAL / STREAM_QUOTA_FAIL
【数值纪律】一律 `.item()` 后取标量比对。
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PKG_DIR = _HERE.parent                     # prototype/
sys.path.insert(0, str(_PKG_DIR))

import torch  # noqa: E402

BACKEND = os.environ.get("DC_BACKEND", "ascend")
OUT_DIR = Path(os.environ.get("DC_OUT_DIR", str(_HERE)))
TAG = os.environ.get("DC_TAG", "")
N = int(os.environ.get("DC_QUOTA_N", "2000"))


def resolve_dev_api() -> str:
    try:
        import runtime  # noqa: E402
        runtime.use(BACKEND)
        dt = getattr(runtime.current(), "device_type", "")
        if dt:
            return dt
    except Exception as exc:
        print(f"[env] runtime 不可用（{type(exc).__name__}），回退环境变量/默认映射")
    if os.environ.get("DC_DEV_API"):
        return os.environ["DC_DEV_API"]
    return {"ascend": "npu", "kunlun": "cuda",
            "cambricon": "mlu"}.get(BACKEND, "cuda")


DEV_API = resolve_dev_api()
dev = getattr(torch, DEV_API)
DEV = DEV_API


def main() -> int:
    print("=== probe_stream_quota.py: S-16 流数量配额（后端无关）===")
    print(f"[env] backend={BACKEND} dev_api={DEV_API} devices={dev.device_count()} "
          f"torch={torch.__version__} N={N}")
    torch.zeros(1, device=DEV)
    dev.set_device(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    checks: dict = {}

    # ── Q1：连续创建 N 个流 ──
    streams = []
    t0 = time.monotonic()
    err = None
    try:
        for _ in range(N):
            streams.append(dev.Stream())
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"
    dt = time.monotonic() - t0
    n_created = len(streams)
    q1_ok = err is None and n_created == N
    print(f"[Q1] 连续创建 {N} 个流：实际成功 {n_created} 个，耗时 {dt:.2f}s "
          f"{'✅' if q1_ok else '❌'}" + (f"  首个失败: {err}" if err else ""))
    checks["Q1_create_n_streams"] = {
        "ok": q1_ok, "detail": f"成功 {n_created}/{N}，耗时 {dt:.2f}s" + (f"，失败: {err}" if err else "")}

    # ── Q2：第 1 个流仍可正常执行 ──
    q2_ok = False
    if streams:
        try:
            with dev.stream(streams[0]):
                a = torch.full((8, 8), 3.0, device=DEV)
                b = a.sum()
            dev.synchronize()
            v = float(b.item())
            q2_ok = abs(v - 192.0) < 1e-3          # 8*8*3
            print(f"[Q2] 第 1 个流仍可执行：结果={v:.4f}（期望 192.0）{'✅' if q2_ok else '❌'}")
            checks["Q2_first_stream_still_works"] = {"ok": q2_ok, "detail": f"结果={v:.4f}（期望 192.0）"}
        except Exception as e:
            print(f"[Q2] 第 1 个流执行异常: {type(e).__name__}: {str(e)[:160]}")
            checks["Q2_first_stream_still_works"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:140]}"}

    # ── Q3：释放后无残留，且能再建新流 ──
    q3_ok = False
    try:
        streams.clear()
        gc.collect()
        dev.synchronize()
        with dev.stream(dev.Stream()):
            c = torch.full((4, 4), 2.0, device=DEV)
            d = c.sum()
        dev.synchronize()
        v3 = float(d.item())
        q3_ok = abs(v3 - 32.0) < 1e-3              # 4*4*2
        print(f"[Q3] 释放 {n_created} 个流后再建新流：结果={v3:.4f}（期望 32.0）{'✅' if q3_ok else '❌'}")
        checks["Q3_release_and_recreate"] = {"ok": q3_ok, "detail": f"释放 {n_created} 个后新流结果={v3:.4f}"}
    except Exception as e:
        print(f"[Q3] 异常: {type(e).__name__}: {str(e)[:160]}")
        checks["Q3_release_and_recreate"] = {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:140]}"}

    passed = sum(1 for v in checks.values() if v.get("ok"))
    total = len(checks)
    verdict = ("STREAM_QUOTA_PASS" if passed == total
               else ("STREAM_QUOTA_PARTIAL" if passed else "STREAM_QUOTA_FAIL"))
    print(f"\n{verdict}: {passed}/{total} 子项通过")

    out = {"verdict": verdict, "passed": passed, "total": total,
           "backend": BACKEND, "dev_api": DEV_API, "n_requested": N,
           "n_created": n_created,
           "torch": torch.__version__, "checks": checks}
    path = OUT_DIR / f"stream_quota_result{TAG}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
