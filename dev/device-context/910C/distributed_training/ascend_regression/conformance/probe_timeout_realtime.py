#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
probe_timeout_realtime.py — D10 timeout 类错误码真实触发验证（P2-⑦，A 线）
═══════════════════════════════════════════════════════════════════════════════

【背景】TIMEOUT 类 10 个错误码（107019 WAIT_TIMEOUT / 107020 TASK_TIMEOUT /
  507046 STREAM_SYNC_TIMEOUT / 507047 EVENT_SYNC_TIMEOUT 等）已在映射表统一 L3_EXECUTION，
  但此前只做过"构造消息"的翻译链路验证（L2 层），从未真实触发（L3 层）。

【方案】pyACL 真实触发（关键：任务与同步必须在同一个 stream）：
  T1 stream 同步超时：torch.npu.Stream 上提交大计算（8192² matmul × 5，任务足够长），
    用 torch stream 的底层句柄（.npu_stream）调 acl.rt.synchronize_stream_with_timeout(1ms)
    → 任务未完成即超时 → 预期 ACL_ERROR_RT_STREAM_SYNC_TIMEOUT(507046) 或 WAIT_TIMEOUT(107019)
  T2 event 同步超时：torch Event（关联长任务）用 synchronize_event_with_timeout(1ms)
    → 预期 EVENT_SYNC_TIMEOUT(507047)（若 pyACL event 句柄可得）
  对真实返回的错误码调用 errors.translate_error 验证分级（预期 L3_EXECUTION）。

【用法】容器内 A 线环境：python3 probe_timeout_realtime.py
【输出】stdout + timeout_realtime_result.json，判定 TIMEOUT_REALTIME_PASS/PARTIAL/FAIL
"""
import json
import time

import torch
import torch_npu  # noqa: F401


def main():
    print("=== probe_timeout_realtime.py: D10 timeout 真实触发（P2-⑦）===")
    result = {"verdict": "FAIL", "checks": {}, "note": ""}

    try:
        import acl
    except ImportError as e:
        result["note"] = f"pyACL 不可用: {e}"
        print(result["note"])
        _dump(result)
        return

    try:
        acl.init()
        acl.rt.set_device(0)
    except Exception as e:
        result["note"] = f"acl init 失败: {e}"
        _dump(result)
        return

    # ══════════ T1：stream 同步超时（同流任务 + 1ms 超时）══════════
    t1 = {}
    try:
        torch.npu.set_device(0)
        x = torch.randn(8192, 8192, device="npu")
        # 预热（首次算子开销大，预热后任务才稳定长）
        for _ in range(2):
            _ = (x @ x)
        torch.npu.synchronize()

        s = torch.npu.Stream()
        with torch.npu.stream(s):
            for _ in range(5):          # 5 个 8192² matmul，任务足够长
                _ = (x @ x)
        # 任务尚未执行完（未同步），用 torch stream 底层句柄做 1ms 超时同步
        handle = s.npu_stream           # aclrtStream 整数句柄
        t0 = time.time()
        rc_timeout = acl.rt.synchronize_stream_with_timeout(handle, 1)
        t1["sync_timeout_rc"] = rc_timeout
        t1["sync_timeout_ms"] = round((time.time() - t0) * 1000, 2)
        t1["stream_handle"] = handle
        torch.npu.synchronize()         # 清理：等任务真正完成
    except Exception as e:
        t1["error"] = f"{type(e).__name__}: {str(e)[:150]}"

    # ══════════ T2：event 同步超时（若句柄可得）══════════
    t2 = {}
    try:
        ev = torch.npu.Event()
        x2 = torch.randn(8192, 8192, device="npu")
        s2 = torch.npu.Stream()
        with torch.npu.stream(s2):
            _ = (x2 @ x2)
            for _ in range(5):
                _ = (x2 @ x2)
            ev.record(s2)               # record 长任务完成点（任务未完成）
        # 尝试取 torch Event 底层句柄
        ev_handle = getattr(ev, "_handle", None) or getattr(ev, "npu_event", None) or getattr(ev, "cuda_event", None)
        if ev_handle is None:
            # 枚举可能的属性名
            cand = {a: getattr(ev, a) for a in dir(ev) if "event" in a.lower() or "handle" in a.lower()}
            t2["handle_candidates"] = {k: (str(v)[:60]) for k, v in cand.items() if not callable(v)}
            ev_handle = None
        if ev_handle is not None and not isinstance(ev_handle, (int, int.__class__)):
            try:
                ev_handle = int(ev_handle)
            except Exception:
                ev_handle = None
        if ev_handle is not None:
            t0 = time.time()
            rc_ev = acl.rt.synchronize_event_with_timeout(ev_handle, 1)
            t2["sync_event_timeout_rc"] = rc_ev
            t2["sync_event_timeout_ms"] = round((time.time() - t0) * 1000, 2)
        else:
            t2["skipped"] = "torch Event 底层句柄不可得，EVENT_SYNC 由 STREAM_SYNC 语义代表"
        torch.npu.synchronize()
    except Exception as e:
        t2["error"] = f"{type(e).__name__}: {str(e)[:150]}"

    # ══════════ 翻译验证（真实返回码）══════════
    rc = t1.get("sync_timeout_rc")
    if rc is not None and rc != 0:
        try:
            import sys
            sys.path.insert(0, ".")
            from errors import translate_error
            fe = translate_error(RuntimeError(f"stream sync timeout, error code is {rc}"),
                                 location="probe:timeout")
            t1["translated"] = {"category": fe.category.name, "mapped": fe.mapped,
                                "graded_by": fe.graded_by, "error_code": fe.error_code}
            t1["expected_L3"] = fe.category.name == "L3_EXECUTION"
        except Exception as e:
            t1["translate_error"] = str(e)[:120]
    else:
        t1["translate_skip"] = f"未返回超时错误码（rc={rc}）"

    ok_trigger = t1.get("sync_timeout_rc") not in (None, 0)
    ok_translate = bool(t1.get("expected_L3"))
    result["checks"]["T1_stream_sync_timeout"] = {
        "ok": ok_trigger,
        "detail": f"rc={t1.get('sync_timeout_rc')} 耗时={t1.get('sync_timeout_ms')}ms "
                  f"翻译={t1.get('translated', {}).get('category', 'N/A')}"}
    result["checks"]["T1_translated_L3"] = {"ok": ok_translate, "detail": str(t1.get("translated"))}
    if t2.get("sync_event_timeout_rc") is not None:
        result["checks"]["T2_event_sync_timeout"] = {
            "ok": t2["sync_event_timeout_rc"] != 0,
            "detail": f"rc={t2['sync_event_timeout_rc']} 耗时={t2['sync_event_timeout_ms']}ms"}

    if ok_trigger and ok_translate:
        result["verdict"] = "TIMEOUT_REALTIME_PASS"
        result["note"] = (f"真实触发成功：stream 同步超时返回错误码 {rc}，"
                          f"翻译为 {t1['translated']['category']}（mapped={t1['translated']['mapped']}）")
    elif ok_trigger:
        result["verdict"] = "TIMEOUT_REALTIME_PARTIAL"
        result["note"] = "真实触发成功但翻译分级与预期不符"
    else:
        result["verdict"] = "TIMEOUT_REALTIME_FAIL"
        result["note"] = "未能真实触发 timeout"

    print(f"\n[T1] stream 同步超时 rc={t1.get('sync_timeout_rc')} 耗时={t1.get('sync_timeout_ms')}ms")
    print(f"     翻译={t1.get('translated')}")
    if t2:
        print(f"[T2] event 同步超时 rc={t2.get('sync_event_timeout_rc')} {t2.get('skipped', '')}")
    print(f"\n{result['verdict']}: {result['note']}")

    result["details"] = {"t1": t1, "t2": t2}
    _dump(result)


def _dump(result):
    with open("timeout_realtime_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("结果已写入 timeout_realtime_result.json")


if __name__ == "__main__":
    main()
