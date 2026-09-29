#!/usr/bin/env python3
"""公开入口「设备状态驱动 + 错误编排」定向验证（**芯片无关**，按 `--backend` 复用）。

═══════════════════════════════════════════════════════════════════════════════
【它守什么】
  2026-09-29 A2 实测发现：`recover_device(mode="real")` **只在设备处于 ISOLATED 时**才真重建，
  而此前**公开面没有把设备置为 ISOLATED 的入口** ⇒ 只走公开 API 时该链**不可触发**。
  本轮补两个只增入口后，本探针把四个方向钉死：

    D1 健康设备 + `mode="real"` ⇒ `context_recreated` 必须 **False**（未重建就不得声称已重建）
    D2 `set_device_state("isolated")` + `mode="real"` ⇒ `recovered=True` **且** `context_recreated=True`
       （真重建必须被如实声明 —— 别把一个缺陷修成另一个方向的缺陷），且重建后回 `available`（R4）
    D3 `set_device_state("isolated")` + `mode="probe"` ⇒ `context_recreated` 必须 **False**
       （探针只重试，不销毁/重建上下文）
    D4 **端到端（公开 API 一次调用）**：置隔离 → `handle_error(<L4 消息>, mode="real")`
       ⇒ `recovery_decision.steps` 出现 `recovered: True` 与 `replay_ready`，设备回 `available`
       —— 这一条直接证明「某卡 L4 故障 → 设备级恢复」这条链**可由公开 API 完整驱动**

  另：全程只用 **公开 API**（`runtime.use` / `set_device_state` / `device_state` /
  `recover_device` / `handle_error`），**不再触碰 `runtime/conformance/` 内部模块** ——
  这也是本轮的目标产物之一。

【边界（勿外推）】
  · ISOLATED 是**构造**出来的（本层账本），**不是真实硬件 L4 故障**；
    "真实故障的触发链"仍不在本探针结论范围内。
  · 结论只在**本档位/环境**成立（镜像、驱动、卡数）；不得外推到其他实例或多机。
  · D4 只在**声明了 `error_map`** 的后端上要求判成 L4（未声明者按码表归属应判 L3 ⇒ 本探针如实跳过 D4）。

【用法】
  DC_ROOT=<prototype> python3 probes/recover_entry_verify.py --backend ascend --dev 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DC_ROOT") or os.path.dirname(_HERE)
sys.path.insert(0, ROOT)

import runtime  # noqa: E402

FIVE = ("ordinal", "mode", "recovered", "state", "detail")
CTX3 = ("context_supported", "context_count", "context_recreated")
TOKENS = tuple(runtime.DEVICE_STATE_TOKENS)
#: 一条**码表内**的 L4 码（昇腾 507015 = AICORE 异常）；未声明 error_map 的后端**不应**据此判 L4。
L4_MSG = "kernel launch failed, error code is 507015"


def _tok(state) -> str:
    """设备状态 → 对外 token（契约：`DeviceState` 的 `.value` 即 token）。"""
    return getattr(state, "value", None) or str(state).split(".")[-1].lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.environ.get("DC_BACKEND", "ascend"))
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rt = runtime.use(args.backend)
    rt.set_device(args.dev)
    dev = args.dev
    checks = {}

    def judge(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail}

    out = {"backend": rt.name, "device_type": rt.device_type, "ordinal": dev,
           "declared_recovery_real": bool(rt.supports("recovery_real")),
           "declared_error_map": bool(rt.supports("error_map")),
           "public_api_only": True}

    # ── D1 健康设备 + real ──
    pre = _tok(rt.device_state(dev))
    r1 = rt.recover_device(dev, mode="real", reason="entry-verify D1")
    post1 = _tok(rt.device_state(dev))
    out["d1"] = {"pre": pre, "post": post1, "return": r1}

    # ── D2 隔离 + real（公开入口驱动）──
    back2 = rt.set_device_state(dev, "isolated", "entry-verify D2: 公开入口构造隔离")
    r2 = rt.recover_device(dev, mode="real", reason="entry-verify D2")
    post2 = _tok(rt.device_state(dev))
    out["d2"] = {"set_returned": back2, "post": post2, "return": r2}

    # ── D3 隔离 + probe ──
    rt.set_device_state(dev, "isolated", "entry-verify D3: 公开入口构造隔离")
    r3 = rt.recover_device(dev, mode="probe", reason="entry-verify D3")
    post3 = _tok(rt.device_state(dev))
    out["d3"] = {"post": post3, "return": r3}

    # ── D4 端到端：一次 handle_error 走完 R1–R5 ──
    rt.set_device_state(dev, "isolated", "entry-verify D4: 公开入口构造隔离")
    fe = rt.handle_error(RuntimeError(L4_MSG), ordinal=dev, location="entry-verify D4", mode="real")
    steps = list((getattr(fe, "recovery_decision", {}) or {}).get("steps", []))
    out["d4"] = {"category": getattr(getattr(fe, "category", None), "name", None),
                 "graded_by": getattr(fe, "graded_by", None),
                 "recovery_decision": getattr(fe, "recovery_decision", None),
                 "post": _tok(rt.device_state(dev))}
    # 收尾：确保设备回 available，别把机器留在隔离态
    try:
        rt.set_device_state(dev, "available", "entry-verify 收尾")
    except Exception as e:                                        # noqa: BLE001
        out["cleanup_error"] = f"{type(e).__name__}: {e}"[:120]

    # ── 判定 ──
    for tag, rec in (("d1", r1), ("d2", r2), ("d3", r3)):
        miss = [k for k in FIVE + CTX3 if k not in rec]
        judge(f"{tag}_keys_5+3", not miss, f"缺 {miss}" if miss else "8 键齐全")
        judge(f"{tag}_state_in_domain", rec.get("state") in TOKENS, f"state={rec.get('state')!r}")

    judge("D1_健康设备real_不得声称已重建", r1.get("context_recreated") is False,
          f"context_recreated={r1.get('context_recreated')!r} detail={r1.get('detail')!r}")
    judge("D2_公开入口置隔离后real_必须声明真重建",
          r2.get("recovered") is True and r2.get("context_recreated") is True,
          f"set 返回={back2!r} recovered={r2.get('recovered')!r} "
          f"context_recreated={r2.get('context_recreated')!r} detail={r2.get('detail')!r}")
    judge("D2_重建后回到 available（R4）", post2 == "available", f"post={post2!r}")
    judge("D3_隔离后probe_不得声称已重建", r3.get("context_recreated") is False,
          f"context_recreated={r3.get('context_recreated')!r} detail={r3.get('detail')!r}")

    if out["declared_error_map"]:
        judge("D4_handle_error 端到端走到 real 重建并重放就绪",
              any(s.startswith("recovered: True") for s in steps) and "replay_ready" in steps,
              f"steps={steps}")
        judge("D4_端到端后设备回 available", out["d4"]["post"] == "available",
              f"post={out['d4']['post']!r}")
    else:
        judge("D4_端到端（本后端未声明 error_map ⇒ 按码表归属**不应**判 L4，如实跳过）",
              out["d4"]["category"] != "L4_FATAL",
              f"category={out['d4']['category']}（若为 L4 说明分级用了外厂码表）")

    allok = all(v["ok"] for v in checks.values())
    out["checks"] = checks
    out["verdict"] = "ENTRY_VERIFY_PASS" if allok else "ENTRY_VERIFY_FAIL"

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print()
    for k, v in checks.items():
        print(f"  {'PASS' if v['ok'] else 'FAIL'}  {k}  {v['detail']}")
    print(f"\n{out['verdict']}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"结果 -> {args.out}")
    print("ENTRY_BOUNDARY: 全程公开 API；ISOLATED 为构造而非真实硬件故障；"
          "结论只在本档位/环境成立，不外推到其他实例或多机。")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
