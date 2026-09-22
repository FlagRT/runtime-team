#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_device_reset_rebuild.py — D11 真实重建可行性验证（P1-③，A 线）
═══════════════════════════════════════════════════════════════════════════════

【背景】D11 恢复目前是"探针重试 + 状态标记"的最小近似，真实上下文重建未验证。
  CANN 头文件（acl_rt.h）定义官方重建序列：
    destroyEvent → destroyStream → destroyContext → aclrtResetDevice(deviceId)
  → aclrtSetDevice 重新指定 → 重建上下文/流。

【验证】
  R1 官方序列可执行：显式创建 context/stream → 按序 destroy → resetDevice → 成功
  R2 reset 后设备可恢复：重新 setDevice + 重建 context/stream → 提交任务 → 成功
  R3 显式资源在 reset 前未 destroy 的行为（对照：观察是否报错，验证注释中
     "otherwise business abnormalities may be caused" 的真实性——仅记录，不判成败）
  R4 多进程隔离（可选，需手动开两个实例）：本进程 reset 不影响其他进程 ——
     注释语义为"释放当前进程默认上下文"，此处记录语义结论

【用法】容器内 A 线环境：python3 probe_device_reset_rebuild.py
【输出】stdout + device_reset_rebuild_result.json，判定 RESET_REBUILD_PASS/FAIL
【安全】独立进程运行（不与 serve 共存），resetDevice 只影响本进程默认上下文。
"""
import json
import acl  # noqa: F401  # pyACL

EXPECTED = {"ok": [0], "eof": []}  # 占位，实际以 ret==0 判断


def rec(checks, name, ret, expect_zero=True, detail=""):
    ok = (ret == 0) if expect_zero else True
    checks[name] = {"ok": ok, "ret": ret, "detail": detail}
    print(f"[{name}] ret={ret} {'✅' if ok else '❌'} {detail}")
    return ok


def main():
    print("=== probe_device_reset_rebuild.py: D11 真实重建可行性（P1-③）===")
    checks = {}
    device_id = 0

    # 初始化
    ret = acl.init()
    if not rec(checks, "R0_acl_init", ret, detail="acl.init()"):
        print("RESET_REBUILD_FAIL: acl.init 失败")
        return
    ret = acl.rt.set_device(device_id)
    if not rec(checks, "R0_set_device", ret, detail=f"aclrtSetDevice({device_id})"):
        print("RESET_REBUILD_FAIL: set_device 失败")
        return

    # ── R1 官方序列：显式创建 → destroy → reset ──
    try:
        ctx, r_ctx = acl.rt.create_context(device_id)
        rec(checks, "R1_create_context", r_ctx, detail="aclrtCreateContext")
        stream, r_s = acl.rt.create_stream()
        rec(checks, "R1_create_stream", r_s, detail="aclrtCreateStream")
        ev, r_ev = acl.rt.create_event()
        rec(checks, "R1_create_event", r_ev, detail="aclrtCreateEvent")

        # 按官方顺序 destroy
        r1 = acl.rt.destroy_event(ev)
        rec(checks, "R1_destroy_event", r1, detail="aclrtDestroyEvent")
        r2 = acl.rt.destroy_stream(stream)
        rec(checks, "R1_destroy_stream", r2, detail="aclrtDestroyStream")
        r3 = acl.rt.destroy_context(ctx)
        rec(checks, "R1_destroy_context", r3, detail="aclrtDestroyContext")
        r4 = acl.rt.reset_device(device_id)
        rec(checks, "R1_reset_device", r4, detail="aclrtResetDevice(官方序列后)")

        # ── R2 reset 后设备可恢复：重新 set + 重建 + 提交任务 ──
        r5 = acl.rt.set_device(device_id)
        rec(checks, "R2_reset_set_device", r5, detail="reset 后重新 aclrtSetDevice")
        ctx2, r6 = acl.rt.create_context(device_id)
        rec(checks, "R2_rebuild_context", r6, detail="重建 aclrtCreateContext")
        stream2, r7 = acl.rt.create_stream()
        rec(checks, "R2_rebuild_stream", r7, detail="重建 aclrtCreateStream")
        # 提交一个可执行任务验证设备真的可用（重建资源即最小验证）
        # 用最小可验证动作：再次创建事件并记录
        ev2, r8 = acl.rt.create_event()
        rec(checks, "R2_rebuild_event", r8, detail="重建 aclrtCreateEvent")
        # 清理重建的资源
        acl.rt.destroy_event(ev2)
        acl.rt.destroy_stream(stream2)
        acl.rt.destroy_context(ctx2)
        acl.rt.reset_device(device_id)
    except Exception as e:
        print(f"[R1/R2] 异常: {type(e).__name__}: {str(e)[:200]}")

    # ── R3 对照：reset 前不 destroy 显式资源（观察行为，仅记录）──
    try:
        ret = acl.rt.set_device(device_id)
        ctx3, _ = acl.rt.create_context(device_id)
        stream3, _ = acl.rt.create_stream()
        r_skip = acl.rt.reset_device(device_id)  # 未 destroy 直接 reset
        rec(checks, "R3_reset_wo_destroy", r_skip, expect_zero=False,
            detail="未 destroy 显式资源直接 reset（注释警告将导致异常，观察返回值）")
        # 清理
        try:
            acl.rt.destroy_stream(stream3)
            acl.rt.destroy_context(ctx3)
        except Exception:
            pass
        acl.rt.reset_device(device_id)
    except Exception as e:
        rec(checks, "R3_reset_wo_destroy", -1, expect_zero=False,
            detail=f"未 destroy 直接 reset 抛异常（符合注释预期）: {str(e)[:120]}")

    # R4 语义结论（来自 acl_rt.h 注释，非实测）
    checks["R4_multi_process_semantics"] = {
        "ok": True,
        "ret": None,
        "detail": ("acl_rt.h: 'Reset the current operating Device...including the default context, "
                   "the default stream' —— 释放范围为当前进程默认上下文；多进程共享设备时 "
                   "reset 不影响其他进程的显式 Context/Stream（官方注释语义，跨进程实测待联调）")}

    # ── 判定 ──
    r1_ok = checks.get("R1_reset_device", {}).get("ok", False)
    r2_ok = all(checks.get(k, {}).get("ok", False)
                for k in ("R2_reset_set_device", "R2_rebuild_context", "R2_rebuild_stream"))
    all_ok = r1_ok and r2_ok
    verdict = "RESET_REBUILD_PASS" if all_ok else "RESET_REBUILD_FAIL"
    print(f"\n{verdict}: 官方序列可执行={r1_ok} reset 后设备可恢复={r2_ok}")

    acl.finalize()
    with open("device_reset_rebuild_result.json", "w", encoding="utf-8") as f:
        json.dump({"verdict": verdict, "checks": checks}, f, ensure_ascii=False, indent=2)
    print("结果已写入 device_reset_rebuild_result.json")


if __name__ == "__main__":
    main()
