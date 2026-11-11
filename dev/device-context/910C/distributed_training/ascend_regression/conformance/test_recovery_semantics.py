#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_recovery_semantics.py — A10「设备状态恢复」语义证伪/证实

【质疑】A10 判定 8/8 PASS，但「恢复」的实质是 recovery.recover_device 里的
      **探针重试成功**（重新跑 probe_device 确认设备可用）即置 AVAILABLE。
      这不是真实的上下文重建（未调用任何 aclrtResetDevice / 销毁重建 context）。
      疑问：这是「重试」还是「恢复」？持久故障下会不会**假恢复**（标记 AVAILABLE 但设备仍不可用）？

【三个场景】
  S1 瞬时故障：前 2 次探针失败、第 3 次成功 → 编排应恢复正常，且恢复后功能可用
  S2 持久故障：探针一直失败 → **必须保持 ISOLATED，绝不可假恢复**（关键证伪点）
  S3 恢复后功能：恢复完成后跑真实 matmul，验证设备确实可用（而非仅状态标记变了）

【另记录】恢复路径是否调用任何真实设备重建 API（用于语义标注：最小近似 vs 真实重建）

【用法】容器内：python3 test_recovery_semantics.py [--ordinal 0]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch
import torch_npu  # noqa: F401

from device_state import DeviceState, query_device_state, set_device_state  # noqa: E402
from recovery import handle_error  # noqa: E402


def make_sync(fail_times):
    """构造可注入故障的 sync_fn；fail_times=999 表示持久故障"""
    calls = {"n": 0}

    def sync():
        calls["n"] += 1
        if calls["n"] <= fail_times:
            raise RuntimeError("inject: device sync failure")
        torch.npu.synchronize()

    return sync, calls


def scenario(fail_times, ordinal, label):
    set_device_state(ordinal, DeviceState.AVAILABLE, reason="scenario reset")
    sync, calls = make_sync(fail_times)
    exc = RuntimeError("NPU fatal error: device lost, context corrupted")
    fe = handle_error(exc, ordinal=ordinal, location=f"device:{ordinal}/op:infer",
                      sync_fn=sync)
    st = query_device_state(ordinal)
    return {
        "label": label,
        "category": fe.category.name,
        "graded_by": getattr(fe, "graded_by", None),
        "mapped": getattr(fe, "mapped", None),
        "recovery_steps": fe.recovery_decision.get("steps"),
        "state_after": st.value,
        "sync_calls": calls["n"],
    }


def post_recovery_function(ordinal):
    """恢复后跑真实计算，验证设备确实可用"""
    try:
        a = torch.randn(512, 512, device=f"npu:{ordinal}")
        b = a @ a
        torch.npu.synchronize()
        ok = bool(torch.isfinite(b).all())
        return {"ok": ok, "detail": "512² matmul 完成且结果有限"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"{type(e).__name__}: {str(e)[:120]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--out", default="recovery_semantics_result.json")
    args = ap.parse_args()

    torch.zeros(1, device=f"npu:{args.ordinal}")  # 预热

    print("=== A10 恢复语义验证 ===")
    s1 = scenario(2, args.ordinal, "S1 瞬时故障(前2次失败)")
    print(f"[S1] category={s1['category']} graded_by={s1['graded_by']} "
          f"state={s1['state_after']} sync_calls={s1['sync_calls']}")
    print(f"     steps={s1['recovery_steps']}")

    s3 = post_recovery_function(args.ordinal)
    print(f"[S3] 恢复后功能: {s3}")

    s2 = scenario(999, args.ordinal, "S2 持久故障(一直失败)")
    print(f"[S2] category={s2['category']} state={s2['state_after']} "
          f"sync_calls={s2['sync_calls']}")
    print(f"     steps={s2['recovery_steps']}")

    # 判定
    transient_ok = (s1["state_after"] == "available" and s3["ok"])
    persistent_ok = (s2["state_after"] == "isolated")  # 关键：不得假恢复

    result = {
        "scenarios": {"S1_transient": s1, "S2_persistent": s2, "S3_post_function": s3},
        "checks": {
            "transient_recovers_and_usable": transient_ok,
            "persistent_stays_isolated_no_fake_recovery": persistent_ok,
        },
        "recovery_is_rebuild": False,
        "semantics_note": (
            "recover_device 的实质是「探针重试确认设备可用后改状态标记」，"
            "未调用任何设备生命周期 API（aclrtResetDevice / context 销毁重建），"
            "属框架层最小近似——设备并未被真正重建，只是被重新判定为可用。"
        ),
        "verdict": "RECOVERY_SEMANTICS_OK" if (transient_ok and persistent_ok)
                   else "RECOVERY_SEMANTICS_PROBLEM",
    }

    print(f"\n=== 判定：{result['verdict']} ===")
    print(f"  瞬时故障可恢复且可用: {transient_ok}")
    print(f"  持久故障不假恢复(保持 ISOLATED): {persistent_ok}   ← 关键证伪点")
    print(f"  恢复是否真实重建: {result['recovery_is_rebuild']}  ({result['semantics_note'][:40]}...)")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果：{args.out}")
    return 0 if result["verdict"] == "RECOVERY_SEMANTICS_OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
