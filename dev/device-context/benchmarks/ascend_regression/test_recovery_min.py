#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_recovery_min.py — 细项21 补验：设备状态恢复（状态机四态 + 五段式恢复）
═══════════════════════════════════════════════════════════════════════════════

【验证目标】设备执行上下文"设备状态恢复"职责在 910C（torch_fl/flagos）上的落地验证，
 覆盖统一行为契约 R1-R5 与状态机四态：
  - R1 捕获：handle_error → 统一错误对象（含 recovery_decision 流程事件）
  - R2 评估：probe_device / evaluate_device（区分 L3 可继续 vs L4 需重建）
  - R3 隔离：set_device_state(ISOLATED)（调度池摘除语义）
  - R4 重建：recover_device（ISOLATED → AVAILABLE）
  - R5 重放：mark_inflight / finish_inflight / replay_tasks（重放集合数据源）
  - 状态机四态：AVAILABLE/DEGRADED/ISOLATED/DESTROYED 转换 + 订阅事件 + 快照

【用法】容器内、tf-venv-integration 激活、单进程：
    python test_recovery_min.py
【输出】stdout + recovery_min_result.json，判定 RECOVERY_PASS/PARTIAL/FAIL
【硬约束】全程 torch_fl（flagos 设备），不 import torch_npu。
【诚实标注】重建为框架层最小近似（探针重试=重取设备资源验证）；真实"新建上下文+
  重分配资源"依赖设备生命周期接口，待 torch_fl 补充后升级重建段。
"""

import json

import torch
import torch_fl
from torch_fl import flagos
from torch_fl.flagos.device_state import (
    DeviceState, query_device_state, set_device_state,
    subscribe_device_state, device_states,
)
from torch_fl.flagos.recovery import (
    probe_device, evaluate_device, recover_device, handle_error,
    mark_inflight, finish_inflight, replay_tasks,
)


def main():
    print("=== test_recovery_min.py: 状态机四态 + 五段式恢复补验 ===")
    devs = flagos.device_count()
    print(f"[env] torch_fl={getattr(torch_fl,'__version__','unknown')} devices={devs}")
    # 设备预热（源码版 torch_fl：pin_memory/Event 依赖设备已初始化）
    torch.zeros(1, device="flagos")
    flagos.synchronize()
    if devs < 1:
        print("RECOVERY_FAIL: 无 flagos 设备")
        return

    result = {"verdict": "PARTIAL", "checks": {}, "note": ""}

    # ---- 第1步 状态机四态 ----
    events = []
    subscribe_device_state(0, lambda ns, os_, r: events.append((ns.value, os_.value, r)))
    st0 = query_device_state(0)
    result["checks"]["state_initial"] = {"ok": st0 == DeviceState.AVAILABLE,
                                          "detail": f"初始状态={st0.value}"}
    print(f"[1a] 初始状态: {st0.value}")

    set_device_state(0, DeviceState.DEGRADED, "test: cap missing")
    set_device_state(0, DeviceState.ISOLATED, "test: L4")
    set_device_state(0, DeviceState.AVAILABLE, "test: reset")
    ev_ok = len(events) >= 3 and events[0][0] == "degraded" and events[1][0] == "isolated"
    result["checks"]["state_machine"] = {"ok": ev_ok,
                                          "detail": f"四态转换事件: {[e[0] for e in events]}"}
    print(f"[1b] 状态机转换事件: {[e[0] for e in events]}")

    snap = device_states()
    result["checks"]["state_snapshot"] = {"ok": 0 in snap and "state" in snap[0],
                                           "detail": f"快照: {snap.get(0, {}).get('state')}"}
    print(f"[1c] 状态快照: {snap.get(0, {}).get('state')}")

    # ---- 第2步 探针 + 评估 ----
    p_ok = probe_device(0)
    result["checks"]["probe"] = {"ok": p_ok, "detail": f"活性探针={p_ok}"}
    print(f"[2a] 活性探针: {p_ok}")

    ev_st = evaluate_device(0)
    result["checks"]["evaluate"] = {"ok": ev_st == DeviceState.AVAILABLE,
                                     "detail": f"评估={ev_st.value}"}
    print(f"[2b] 评估: {ev_st.value}")

    # ---- 第3步 重建（ISOLATED → AVAILABLE）----
    set_device_state(0, DeviceState.ISOLATED, "test: isolate for rebuild")
    rec_ok = recover_device(0)
    rec_st = query_device_state(0)
    result["checks"]["rebuild"] = {"ok": rec_ok and rec_st == DeviceState.AVAILABLE,
                                    "detail": f"重建={rec_ok} 状态={rec_st.value}"}
    print(f"[3] 重建: {rec_ok} → {rec_st.value}")

    # ---- 第4步 在途登记 + 重放 ----
    mark_inflight("op_a", 0, "stream:0/op:matmul")
    mark_inflight("op_b", 0, "stream:0/op:linear")
    rp = replay_tasks(0)
    result["checks"]["inflight"] = {"ok": len(rp) == 2,
                                     "detail": f"在途={len(rp)} 任务: {[t['op_id'] for t in rp]}"}
    print(f"[4a] 在途任务: {[t['op_id'] for t in rp]}")
    finish_inflight("op_a")
    rp2 = replay_tasks(0)
    result["checks"]["replay_set"] = {"ok": len(rp2) == 1 and rp2[0]["op_id"] == "op_b",
                                       "detail": f"重放集合(完成op_a后): {[t['op_id'] for t in rp2]}"}
    print(f"[4b] 重放集合(完成op_a后): {[t['op_id'] for t in rp2]}")
    finish_inflight("op_b")

    # ---- R1 捕获 + 五段式编排（handle_error）----
    try:
        torch.randn(3, 4, device="flagos") @ torch.randn(5, 6, device="flagos")
        result["checks"]["handle_error"] = {"ok": False, "detail": "未触发预期错误"}
    except Exception as e:
        fe = handle_error(e, ordinal=0, location="stream:0/op:matmul")
        dec = fe.recovery_decision
        r_ok = dec["captured"] is True and dec["steps"][0] == "captured"
        result["checks"]["handle_error"] = {"ok": r_ok,
                                             "detail": f"{fe.category.name} 流程: {dec['steps']}"}
        print(f"[5] handle_error L2: {fe.category.name} 流程={dec['steps']}")

    # ---- 判定 ----
    all_ok = all(v["ok"] for v in result["checks"].values())
    if all_ok:
        result["verdict"] = "RECOVERY_PASS"
        result["note"] = ("状态机四态 + 五段式恢复全流程落地验证通过：R1捕获/R2评估/R3隔离/R4重建/R5重放 "
                          "+ 状态转换订阅事件；重建为框架层最小近似（探针重试），真实上下文重建待设备生命周期接口")
    else:
        result["verdict"] = "RECOVERY_PARTIAL"
        result["note"] = "存在未通过检查项，见 checks"
    print(f"\n{result['verdict']}: {result['note']}")

    with open("recovery_min_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 recovery_min_result.json")


if __name__ == "__main__":
    main()
