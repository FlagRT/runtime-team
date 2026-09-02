#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
test_rebuild_multiprocess.py — R4 真实重建多进程联调（P1-③ 落地验证，A 线）
═══════════════════════════════════════════════════════════════════════════════

【背景】recovery.py 的 recover_device 新增 rebuild_mode="real"（CANN 官方
  aclrtResetDevice 序列）。本脚本做多进程本地联调，验证三个关键点：

  S1 恢复者进程（B）真实重建成功：handle_error(L4) → recover_device(real) → recovered=True
  S2 隔离性（官方语义）：B 执行 aclrtResetDevice 期间/之后，同卡另一进程（A，
    模拟长驻服务）持续计算不受影响（多进程共享设备时 reset 只作用于当前进程）
  S3 B 重建后可恢复：reset 后重新 set_device + 重建句柄（pyACL event/stream），
    以及 torch_npu 重新 set_device 后计算恢复

【用法】容器内 A 线环境（双进程共享 NPU 0）：
  python3 test_rebuild_multiprocess.py [--ordinal 0] [--rounds 30]
【输出】stdout + rebuild_multiprocess_result.json，判定 MULTIPROC_REBUILD_PASS/FAIL
【硬约束】A 线 torch_npu；multiprocessing spawn（fork 在 NPU 环境有风险）。
"""
import argparse
import json
import multiprocessing as mp
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu"


def run_server(ordinal: int, rounds: int, q: mp.Queue, evt_b_reset: mp.Event):
    """进程 A：长驻服务模拟——持续设备计算，全程不受 B 的 reset 影响。"""
    try:
        torch.npu.set_device(ordinal)
        x = torch.randn(1024, 1024, device=DEV)
        ok_runs = 0
        fails = []
        for i in range(rounds):
            try:
                y = (x @ x).sum()
                torch.npu.synchronize()
                _ = y.item()
                ok_runs += 1
            except Exception as e:
                fails.append((i, str(e)[:80]))
                # 尝试恢复服务自身
                try:
                    torch.npu.set_device(ordinal)
                    x = torch.randn(1024, 1024, device=DEV)
                except Exception:
                    pass
            if i == rounds // 3:
                evt_b_reset.set()   # 通知 B 可以开始 reset（A 已稳定运行）
            time.sleep(0.1)
        q.put({"ok_runs": ok_runs, "fails": fails, "total": rounds})
    except Exception as e:
        q.put({"ok_runs": 0, "fails": [("init", str(e)[:100])], "total": rounds})


def run_recoverer(ordinal: int, q: mp.Queue, evt_b_reset: mp.Event):
    """进程 B：恢复者——等待 A 稳定后，触发 L4 并执行真实重建。"""
    import sys
    sys.path.insert(0, ".")
    from device_state import DeviceState
    from recovery import handle_error
    from errors import ErrorCategory

    result = {}
    try:
        torch.npu.set_device(ordinal)
        # 模拟 B 也是设备上的工作进程（有自己的上下文）
        x = torch.randn(256, 256, device=DEV)
        (x @ x).sum().item()
        result["b_init_ok"] = True
    except Exception as e:
        result["b_init_ok"] = False
        result["b_init_err"] = str(e)[:100]

    evt_b_reset.wait(timeout=120)   # 等 A 稳定

    # 触发 L4 错误 → handle_error（观察：健康设备上 evaluate 探针必过，不进入重建）
    try:
        fe = handle_error(RuntimeError("device lost simulation"), ordinal=ordinal,
                          location="mp:recoverer", device="npu")
        result["handle_error_decision"] = fe.recovery_decision
        result["category"] = fe.category.name
    except Exception as e:
        result["handle_error_decision"] = {"error": str(e)[:150]}

    # R4 真实重建路径：显式模拟 R3 隔离（设备上下文判定损坏 → ISOLATED），
    # 然后 recover_device(rebuild_mode="real") 执行真实 aclrtResetDevice 序列。
    # 注：健康设备下 handle_error 不会主动隔离（evaluate 探针必过），
    # 真实故障时 R3 由设备状态判定置 ISOLATED，此处等价模拟。
    try:
        from device_state import set_device_state
        set_device_state(ordinal, DeviceState.ISOLATED,
                         "mp:recoverer simulated R3 isolation")
        from recovery import recover_device
        ok = recover_device(ordinal, rebuild_mode="real",
                            reason="mp:recoverer real rebuild")
        result["recovered"] = bool(ok)
        result["recovery_decision"] = {
            "steps": ["captured", "evaluated: (simulated L4)", "isolated",
                      f"recovered: {ok} (real aclrtResetDevice)"]}
    except Exception as e:
        result["recovered"] = False
        result["recovery_decision"] = {"steps": [f"error: {str(e)[:150]}"]}

    # S3：重建后恢复验证——pyACL 重建句柄 + torch_npu 重新 set_device 计算
    try:
        import acl
        acl.init()
        rc = acl.rt.set_device(ordinal)
        stream, r2 = acl.rt.create_stream()
        ev, r3 = acl.rt.create_event()
        result["s3_acl_rebuild"] = {"set_device": rc, "create_stream": r2, "create_event": r3}
        acl.rt.destroy_event(ev)
        acl.rt.destroy_stream(stream)
        acl.rt.reset_device(ordinal)
        # 不显式 finalize：torch_npu 进程退出时会处理，避免 repeated deinit 告警
    except Exception as e:
        result["s3_acl_rebuild"] = {"error": str(e)[:120]}

    try:
        torch.npu.set_device(ordinal)
        y = (torch.randn(512, 512, device=DEV) @ torch.randn(512, 512, device=DEV)).sum()
        torch.npu.synchronize()
        result["s3_torch_recover"] = {"ok": True, "val": round(y.item(), 4)}
    except Exception as e:
        result["s3_torch_recover"] = {"ok": False, "error": str(e)[:120]}
    q.put(result)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ordinal", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=30)
    args = ap.parse_args()

    print(f"=== test_rebuild_multiprocess.py: R4 真实重建多进程联调（device {args.ordinal}）===")
    print(f"[env] torch_npu={getattr(torch_npu, '__version__', 'unknown')} devices={torch.npu.device_count()}")
    torch.zeros(1, device=DEV)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    evt = ctx.Event()
    pa = ctx.Process(target=run_server, args=(args.ordinal, args.rounds, q, evt), name="server-A")
    pb = ctx.Process(target=run_recoverer, args=(args.ordinal, q, evt), name="recoverer-B")
    pa.start(); pb.start()
    pa.join(timeout=args.rounds * 0.2 + 60)
    pb.join(timeout=args.rounds * 0.2 + 60)

    res = {"ordinal": args.ordinal, "rounds": args.rounds, "server": {}, "recoverer": {}}
    # 收集（可能乱序）
    for _ in range(2):
        try:
            item = q.get(timeout=5)
            if "ok_runs" in item:
                res["server"] = item
            else:
                res["recoverer"] = item
        except Exception:
            pass

    sv, rb = res["server"], res["recoverer"]
    print(f"\n[进程A 服务] 成功 {sv.get('ok_runs')}/{sv.get('total')} 次, 失败 {len(sv.get('fails', []))} 次")
    for f in sv.get("fails", [])[:3]:
        print(f"          失败@{f[0]}: {f[1]}")
    print(f"[进程B 恢复] init_ok={rb.get('b_init_ok')} recovered={rb.get('recovered')}")
    print(f"          decision={rb.get('recovery_decision', {}).get('steps', [])}")
    print(f"          S3 acl 重建={rb.get('s3_acl_rebuild')}")
    print(f"          S3 torch 恢复={rb.get('s3_torch_recover')}")

    # ── 判定 ──
    s1 = bool(rb.get("recovered"))
    s2 = sv.get("ok_runs", 0) == sv.get("total", 0) or len(sv.get("fails", [])) == 0
    s3a = (rb.get("s3_acl_rebuild") or {}).get("set_device") == 0
    s3b = bool((rb.get("s3_torch_recover") or {}).get("ok"))
    all_ok = s1 and s2 and s3a and s3b
    res["checks"] = {
        "S1_recoverer_real_rebuild": {"ok": s1, "detail": f"steps={rb.get('recovery_decision', {}).get('steps', [])}"},
        "S2_server_isolation": {"ok": s2, "detail": f"服务进程 {sv.get('ok_runs')}/{sv.get('total')} 轮无失败（B reset 期间不受影响）"},
        "S3a_rebuild_acl_handles": {"ok": s3a, "detail": str(rb.get('s3_acl_rebuild'))},
        "S3b_recoverer_torch_recover": {"ok": s3b, "detail": str(rb.get('s3_torch_recover'))},
    }
    res["verdict"] = "MULTIPROC_REBUILD_PASS" if all_ok else "MULTIPROC_REBUILD_FAIL"
    print(f"\n{res['verdict']}: S1={s1} S2={s2} S3a={s3a} S3b={s3b}")

    with open("rebuild_multiprocess_result.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入 rebuild_multiprocess_result.json")


if __name__ == "__main__":
    main()
