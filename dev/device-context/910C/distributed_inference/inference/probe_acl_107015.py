#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_acl_107015.py — P3 · A9 验收：ACL 107015 真实错误注入 + 错误码翻译验证

【背景】ACL_ERROR_RT_STREAM_NO_CB_REG = 107015
       定义见 /usr/local/Ascend/ascend-toolkit/latest/include/acl/error_codes/rt_error_codes.h
         #define ACL_ERROR_RT_STREAM_NO_CB_REG 107015 // callback not register to stream
       语义：对未订阅 callback 的 stream 直接投递 callback → 契约违反

【职责对应】D10 错误码翻译（F1 三维翻译 / F2 分级处置 / F4 根因保留）
【实验设计】最小变更 + 单变量隔离：
       A 组（错误路径）：create_stream → 直接 launch_callback（不 subscribe）→ 预期 107015
       B 组（正确对照）：create_stream → subscribe_report → launch_callback → 预期 0
       唯一变量 = 是否 subscribe_report
【翻译验证】把 A 组错误喂给 conformance/errors.py 的 translate_error，
            记录当前分级结果，并给出映射表是否需补 107015 的结论
【用法】容器内：python3 probe_acl_107015.py [--device 0] [--out ...]
"""
import argparse
import ctypes
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.join(HERE, "..", "ascend_regression", "conformance")

EXPECTED_CODE = 107015
CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _noop(arg):  # noqa: ARG001
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", default="acl_107015_result.json")
    args = ap.parse_args()

    result = {
        "verdict": "ACL_107015_FAIL",
        "expected_code": EXPECTED_CODE,
        "device": args.device,
        "steps": [],
        "groups": {},
        "translation": {},
        "note": "",
    }

    try:
        import acl
    except Exception as e:  # noqa: BLE001
        result["note"] = f"pyACL 导入失败: {e}"
        print(f"[FAIL] {result['note']}")
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return 1

    def step(name, ret, extra=None):
        rec = {"step": name, "ret": ret}
        if extra:
            rec.update(extra)
        result["steps"].append(rec)
        print(f"  [{name}] ret={ret}" + (f" {extra}" if extra else ""))
        return ret

    print("[1/5] pyACL 初始化")
    ret = acl.init()
    step("acl.init", ret)
    if ret != 0:
        result["note"] = f"acl.init 失败 ret={ret}"
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return 1

    dev = args.device
    ret = acl.rt.set_device(dev)
    step("acl.rt.set_device", ret, {"device": dev})
    if ret != 0:
        # 兜底：换一张卡（serve 可能独占 device 0）
        for alt in (1, 2, 3, 4, 5, 6, 7):
            ret = acl.rt.set_device(alt)
            step(f"acl.rt.set_device(fallback {alt})", ret, {"device": alt})
            if ret == 0:
                dev = alt
                result["device"] = dev
                break
        if ret != 0:
            result["note"] = "set_device 全部失败"
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return 1

    cb = CB(_noop)

    # ── A 组：错误路径（不订阅直接投递）──
    print("[2/5] A 组：create_stream → 直接 launch_callback（不 subscribe）")
    stream_a, ret = acl.rt.create_stream()
    step("A:create_stream", ret)
    # 实测签名：launch_callback(fn, userData, block, stream) —— 四参数，缺一即 args parse failed
    ret_a = acl.rt.launch_callback(cb, None, 0, stream_a) if ret == 0 else ret
    step("A:launch_callback(no subscribe)", ret_a, {"expected": EXPECTED_CODE})
    result["groups"]["A_no_subscribe"] = {
        "launch_callback_ret": ret_a,
        "triggered_107015": ret_a == EXPECTED_CODE,
    }

    # ── B 组：正确对照（先订阅再投递）──
    print("[3/5] B 组：create_stream → subscribe_report → launch_callback")
    stream_b, ret = acl.rt.create_stream()
    step("B:create_stream", ret)
    tid = threading.current_thread().ident or 0
    ret_sub = acl.rt.subscribe_report(tid, stream_b) if ret == 0 else ret
    step("B:subscribe_report", ret_sub, {"thread_id": tid})
    ret_b = (
        acl.rt.launch_callback(cb, None, 0, stream_b) if ret_sub == 0 else None
    )
    step("B:launch_callback(after subscribe)", ret_b, {"expected": 0})
    result["groups"]["B_after_subscribe"] = {
        "subscribe_report_ret": ret_sub,
        "launch_callback_ret": ret_b,
        "success": ret_b == 0,
    }

    # ── 翻译验证 ──
    print("[4/5] 用 conformance/errors.py 翻译 A 组错误")
    sys.path.insert(0, os.path.abspath(CONF_DIR))
    try:
        from errors import translate_error  # type: ignore

        # 复刻 torch_npu/op-plugin 形态的错误消息（errors.py 的两种兼容形态之一）
        msg_a = f"ACL runtime error: launch_callback failed, error code is {ret_a}"
        exc = RuntimeError(msg_a)
        fe = translate_error(exc, location=f"device:{dev}/stream:A/op:launch_callback")
        result["translation"] = {
            "error_code": fe.error_code,
            "category": fe.category.name,
            "category_value": int(fe.category),
            "location": fe.location,
            "root_cause": fe.root_cause,
            "is_retryable": fe.is_retryable,
            "is_fatal": fe.is_fatal,
            "in_acl_map": fe.error_code in _acl_map_codes(),
        }
        print(f"  错误码={fe.error_code} 类别={fe.category.name} 可重试={fe.is_retryable}")
    except Exception as e:  # noqa: BLE001
        result["translation"] = {"error": str(e)}
        print(f"  翻译失败: {e}")

    # ── 判定 ──
    print("[5/5] 判定")
    triggered = result["groups"]["A_no_subscribe"]["triggered_107015"]
    control_ok = result["groups"]["B_after_subscribe"]["success"]
    translated = result["translation"].get("category") is not None
    result["checks"] = {
        "a_triggered_107015": triggered,
        "b_control_success": control_ok,
        "translated": translated,
    }
    ok = triggered and translated
    result["verdict"] = "ACL_107015_PASS" if ok else "ACL_107015_FAIL"
    if triggered and not control_ok:
        result["note"] += "A 组触发成功但 B 组对照未成功（subscribe 路径需另行确认）；"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 判定：{result['verdict']} ===")
    print(f"A 组触发 107015: {triggered} (ret={ret_a}) | B 组对照成功: {control_ok} (ret={ret_b})")
    print(
        f"翻译: {result['translation'].get('category')} "
        f"(错误码 {result['translation'].get('error_code')}, "
        f"在映射表中: {result['translation'].get('in_acl_map')})"
    )
    print(f"结果：{args.out}")
    return 0 if ok else 1


def _acl_map_codes():
    try:
        from errors import ACL_ERR_TO_CATEGORY  # type: ignore

        return set(ACL_ERR_TO_CATEGORY.keys())
    except Exception:  # noqa: BLE001
        return set()


if __name__ == "__main__":
    raise SystemExit(main())
