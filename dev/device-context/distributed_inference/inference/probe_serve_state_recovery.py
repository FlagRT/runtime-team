#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_serve_state_recovery.py — P3 · A10 验收：长驻服务四态监控 + 五段式恢复

【职责对应】D11 设备状态恢复（R1 捕获 → R2 评估 → R3 隔离 → R4 重建 → R5 重放）
【验收标准】A10：四态查询可用 + 注入错误可恢复 + 服务继续可用

【重要机制（实测确认）】recovery.handle_error 只对 **L4_FATAL** 触发 R2-R5 设备恢复流程；
  L1-L3 直接返回，仅标记 replayable（R2 的设计意图：避免不必要的高代价重建）。
  因此本探针覆盖两条分支：
    · L4 分支：构造 device-lost 类错误 → 观察 captured→evaluated→(isolated)→recovered→replay_ready
    · L3 分支：107015（O4 实测真实错误）→ 验证**不触发**重建，仅走重放
    另加状态机闭环演练：手动置 ISOLATED → recover_device → 回 AVAILABLE

【用法】容器内：python3 probe_serve_state_recovery.py [--port 8100] [--ordinal 0] [--out ...]
【判定】DEVICE_STATE_RECOVERY_PASS = 四态可查 + 转换生效 + 两条分支行为正确 + 恢复闭环 + 服务续跑
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.abspath(os.path.join(HERE, "..", "ascend_regression", "conformance"))
sys.path.insert(0, CONF_DIR)


def http_post(port, payload, timeout=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--out", default="serve_state_recovery_result.json")
    args = ap.parse_args()

    result = {
        "verdict": "DEVICE_STATE_RECOVERY_FAIL",
        "ordinal": args.ordinal,
        "phases": {},
        "checks": {},
        "note": "",
    }

    from device_state import (  # noqa: E402
        DeviceState,
        query_device_state,
        set_device_state,
        subscribe_device_state,
    )
    from recovery import handle_error, recover_device  # noqa: E402

    def sync_fn():
        import torch_npu  # noqa: F401
        import torch

        torch.npu.synchronize()

    # 订阅回调签名不固定（不同版本参数个数不同），用 *a 兜底并统一转可序列化类型
    events = []

    def _on_state_change(*a):
        events.append(
            [x.value if hasattr(x, "value") else str(x) for x in a]
        )

    subscribe_device_state(args.ordinal, _on_state_change)

    # ── 阶段 1：四态可查 ──
    print("[1/5] 四态查询（serve 长驻中）")
    st0 = query_device_state(args.ordinal)
    result["phases"]["initial_state"] = st0.value
    print(f"   初始状态: {st0.value}")

    # ── 阶段 2：并发压力注入 + DEGRADED 转换 ──
    print(f"[2/5] 压力注入：{args.concurrency} 并发请求 + DEGRADED 状态转换")
    latencies = []
    errs = []

    def worker(i):
        t0 = time.time()
        try:
            http_post(
                args.port,
                {
                    "model": "qwen3-4b",
                    "prompt": f"请简要说明第 {i} 个主题：",
                    "max_tokens": 64,
                    "temperature": 0,
                },
            )
            latencies.append(round(time.time() - t0, 2))
        except Exception as e:  # noqa: BLE001
            errs.append(str(e)[:100])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = round(time.time() - t0, 2)

    set_device_state(args.ordinal, DeviceState.DEGRADED, reason="并发压力注入演练")
    st_deg = query_device_state(args.ordinal)
    set_device_state(args.ordinal, DeviceState.AVAILABLE, reason="压力解除")
    st_back = query_device_state(args.ordinal)

    result["phases"]["stress"] = {
        "concurrency": args.concurrency,
        "wall_s": wall,
        "latencies": latencies,
        "errors": errs,
        "state_after_degrade": st_deg.value,
        "state_after_restore": st_back.value,
        "state_events": events[-6:],
    }
    print(f"   并发 {args.concurrency} 完成 {wall}s，延迟 {latencies}")
    print(f"   DEGRADED 转换: {st_deg.value} → 恢复: {st_back.value}")

    # ── 阶段 3：L4 分支（真实健康设备 → 应评估为可用、不重建）──
    print("[3/5] L4 分支：device-lost 类错误 → handle_error（R2 应避免不必要重建）")
    exc_l4 = RuntimeError(
        "NPU function error: device lost, context corrupted, error code is 507021"
    )
    fe_l4 = handle_error(
        exc_l4, ordinal=args.ordinal, location=f"device:{args.ordinal}/op:infer",
        sync_fn=sync_fn,
    )
    result["phases"]["l4_branch"] = {
        "category": fe_l4.category.name,
        "error_code": fe_l4.error_code,
        "recovery_decision": getattr(fe_l4, "recovery_decision", {}),
        "state_after": query_device_state(args.ordinal).value,
    }
    print(f"   类别={fe_l4.category.name} 码={fe_l4.error_code} "
          f"steps={getattr(fe_l4, 'recovery_decision', {}).get('steps')}")

    # ── 阶段 3b：L4 + 瞬时故障注入（验证完整 R1→R5）──
    # 说明：健康设备下 evaluate_device 探针必然通过 → 永不触发 R3/R4（R2 设计意图）。
    #       故用瞬时故障 sync（前 2 次失败、第 3 次成功）模拟"故障自愈"，
    #       才能走通 isolated → recovered → replay_ready 完整链路。
    print("[3b/5] L4 完整五段式：纯 L4 消息 + 瞬时故障 sync（前 2 次失败）")
    fail_n = {"n": 0}

    def flaky_sync():
        fail_n["n"] += 1
        if fail_n["n"] <= 2:
            raise RuntimeError("inject: transient device sync failure")
        import torch

        torch.npu.synchronize()

    exc_l4b = RuntimeError("NPU fatal error: device lost, context corrupted")
    fe_l4b = handle_error(
        exc_l4b, ordinal=args.ordinal,
        location=f"device:{args.ordinal}/op:infer", sync_fn=flaky_sync,
    )
    steps_b = getattr(fe_l4b, "recovery_decision", {}).get("steps", [])
    result["phases"]["l4_full_r1_r5"] = {
        "category": fe_l4b.category.name,
        "recovery_decision": getattr(fe_l4b, "recovery_decision", {}),
        "state_after": query_device_state(args.ordinal).value,
        "sync_calls": fail_n["n"],
    }
    print(f"   类别={fe_l4b.category.name} steps={steps_b}")

    # ── 阶段 4：L3 分支（107015 真实错误 → 应不触发重建，仅重放）──
    print("[4/5] L3 分支：107015（O4 实测真实错误）→ 应不触发设备重建")
    exc_107015 = RuntimeError(
        "ACL runtime error: launch_callback failed, error code is 107015"
    )
    fe_107015 = handle_error(
        exc_107015, ordinal=args.ordinal,
        location=f"device:{args.ordinal}/stream:A/op:launch_callback",
        sync_fn=sync_fn,
    )
    result["phases"]["l3_branch"] = {
        "category": fe_107015.category.name,
        "error_code": fe_107015.error_code,
        "recovery_decision": getattr(fe_107015, "recovery_decision", {}),
        "state_after": query_device_state(args.ordinal).value,
    }
    print(f"   类别={fe_107015.category.name} 码={fe_107015.error_code} "
          f"steps={getattr(fe_107015, 'recovery_decision', {}).get('steps')}")

    # ── 阶段 5：状态机闭环演练 ISOLATED → recover → AVAILABLE ──
    print("[5/5] 状态机闭环演练：手动 ISOLATED → recover_device → AVAILABLE")
    set_device_state(args.ordinal, DeviceState.ISOLATED, reason="恢复演练：模拟隔离")
    st_iso = query_device_state(args.ordinal)
    ok_rec = recover_device(args.ordinal, reason="恢复演练", sync_fn=sync_fn)
    st_final = query_device_state(args.ordinal)

    # 服务续跑验证
    serve_ok = False
    serve_text = ""
    try:
        r = http_post(
            args.port,
            {"model": "qwen3-4b", "prompt": "The capital of France is",
             "max_tokens": 16, "temperature": 0},
        )
        serve_text = r["choices"][0]["text"][:80]
        serve_ok = len(serve_text.strip()) > 0
    except Exception as e:  # noqa: BLE001
        result["note"] += f"服务续跑失败: {e}；"

    result["phases"]["recovery_drill"] = {
        "state_isolated": st_iso.value,
        "recover_ok": ok_rec,
        "state_final": st_final.value,
        "serve_after_recovery_ok": serve_ok,
        "serve_text": serve_text,
    }
    print(f"   ISOLATED={st_iso.value} → recover={ok_rec} → {st_final.value}")
    print(f"   服务续跑: {serve_ok} {serve_text[:40]!r}")

    # ── 判定 ──
    result["checks"] = {
        "four_state_queryable": st0 == DeviceState.AVAILABLE,
        "degrade_transition": st_deg == DeviceState.DEGRADED,
        "restore_transition": st_back == DeviceState.AVAILABLE,
        "l4_evaluated": "evaluated" in str(getattr(fe_l4, "recovery_decision", {})),
        "l4_full_r1_r5": (
            fe_l4b.category.name == "L4_FATAL"
            and any("recovered: True" in str(s) for s in steps_b)
            and any("replay_ready" in str(s) for s in steps_b)
            and query_device_state(args.ordinal) == DeviceState.AVAILABLE
        ),
        "l3_no_rebuild": "recovered" not in str(
            getattr(fe_107015, "recovery_decision", {})
        ),
        "isolated_to_available": st_iso == DeviceState.ISOLATED
        and st_final == DeviceState.AVAILABLE,
        "serve_after_recovery": serve_ok,
    }
    ok = all(result["checks"].values())
    result["verdict"] = "DEVICE_STATE_RECOVERY_PASS" if ok else "DEVICE_STATE_RECOVERY_FAIL"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 判定：{result['verdict']} ===")
    for k, v in result["checks"].items():
        print(f"   {k}: {v}")
    print(f"结果：{args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
