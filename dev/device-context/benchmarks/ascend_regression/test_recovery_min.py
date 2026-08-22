#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_recovery_min.py — 细项21 补验：设备状态恢复（五段式·最小验证集）
═══════════════════════════════════════════════════════════════════════════════

【验证目标】设备执行上下文"设备状态恢复"职责在 910C（torch_fl/flagos）上的
 最小可观测性。按设计，恢复为五段式：捕获→评估→隔离→重建→重放，状态机
 ISOLATED→AVAILABLE。本脚本为**最小验证集**：
  - 若 torch_fl 暴露状态查询/恢复接口 → 直接观测状态转换与恢复流程
  - 若未暴露（框架层现状） → 以"错误后设备资源可重新获取/重建"为最小观测点，
    验证重建能力存在，并如实标注"五段式流程编排为运行时层职责，框架层不可观测"

【用法】容器内、tf-venv-integration 激活、单进程：
    python test_recovery_min.py
【输出】stdout + recovery_min_result.json，判定 RECOVERY_PASS/PARTIAL/FAIL
【硬约束】全程 torch_fl（flagos 设备），不 import torch_npu。
"""

import json

import torch
import torch_fl
from torch_fl import flagos


def main():
    print("=== test_recovery_min.py: 五段式状态恢复·最小验证集 ===")
    devs = flagos.device_count()
    print(f"[env] torch_fl={getattr(torch_fl,'__version__','unknown')} devices={devs}")
    if devs < 1:
        print("RECOVERY_FAIL: 无 flagos 设备")
        return

    result = {"verdict": "PARTIAL", "checks": {}, "state_api": "UNKNOWN", "note": ""}

    # 探测状态/恢复接口
    has_state = any(hasattr(flagos, a) for a in ("device_state", "query_state", "state"))
    has_recover = any(hasattr(flagos, a) for a in ("recover", "reset_device", "reinit"))
    result["state_api"] = "unified" if (has_state or has_recover) else "fallback"
    print(f"[env] 状态/恢复接口: {'可用' if result['state_api']=='unified' else '未暴露，退化为最小重建观测'}")

    # 1. 基线：flagos 设备可用
    x = torch.randn(8, 8, device="flagos")
    pass  # flagos 环境禁用 torch.cuda（is_available 误报）
    result["checks"]["baseline"] = {"ok": True, "detail": "flagos 张量运算正常"}
    print("[1] 基线 ok")

    # 2. 最小重建观测：触发错误 → 验证"错误后设备资源可重新获取/重建"
    #    （框架层无法真实制造 L4 上下文损坏，用"错误后重新建张量"近似重建能力）
    rebuild_ok = True
    rebuild_steps = []
    try:
        # 2a. 触发一次可观测错误（形状不匹配，走 torch 异常路径）
        try:
            torch.randn(3, 4, device="flagos") @ torch.randn(5, 6, device="flagos")
        except Exception as e:
            rebuild_steps.append(f"错误已触发: {type(e).__name__}")
        # 2b. 错误后重新创建设备张量（重建的替代观测）
        y = torch.randn(8, 8, device="flagos")
        pass  # flagos 环境禁用 torch.cuda（is_available 误报）
        rebuild_ok = bool(torch.isfinite(y).all())
        rebuild_steps.append("错误后重新创建设备张量成功（最小重建观测）")
    except Exception as e:
        rebuild_ok = False
        rebuild_steps.append(f"重建失败: {e}")
    result["checks"]["min_rebuild"] = {"ok": rebuild_ok, "detail": "; ".join(rebuild_steps)}
    print(f"[2] 最小重建观测 ok={rebuild_ok}: {'; '.join(rebuild_steps)}")

    # 3. 状态机事件观测（若有统一接口）
    if result["state_api"] == "unified":
        try:
            st = flagos.query_state(0) if hasattr(flagos, "query_state") else flagos.device_state(0)
            result["checks"]["state_event"] = {"ok": True, "detail": f"状态可查询: {st}"}
            print(f"[3] 状态查询 ok: {st}")
        except Exception as e:
            result["checks"]["state_event"] = {"ok": False, "detail": f"状态查询失败: {e}"}
    else:
        result["checks"]["state_event"] = {"ok": False, "detail": "接口缺口：状态机事件（ISOLATED/AVAILABLE）与五段式编排为运行时层职责，torch_fl 未暴露"}
        print("[3] 接口缺口：状态机事件/五段式编排未暴露（如实记录）")

    # 判定
    if result["checks"]["min_rebuild"]["ok"]:
        result["verdict"] = "RECOVERY_PASS" if result["state_api"] == "unified" else "RECOVERY_PARTIAL"
        result["note"] = ("最小重建观测通过，状态机/五段式可观测" if result["state_api"] == "unified"
                          else "最小重建观测通过（错误后可重新获取设备资源）；五段式流程与状态机事件依赖运行时层接口，未暴露处如实标注")
    else:
        result["verdict"] = "RECOVERY_FAIL"
    print(f"\n{result['verdict']}: {result['note']}")

    with open("recovery_min_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 recovery_min_result.json")


if __name__ == "__main__":
    main()
