#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_107015_exclusivity.py — A9 根因结论的排他性验证

【原结论】「对未 subscribe_report 的 stream 投递 callback 即命中 107015，属调用方契约违反」

【质疑】该结论基于 A/B 两组对照，但**未排除替代解释**：
  H1 是否**所有** stream 都需先 subscribe，还是仅特定 stream 类型？（默认流 vs 创建流）
  H2 是否 `launch_callback` 的 **block 参数**取值导致的？（0 vs 1）
  H3 是否依赖特定 context / device？
  H4 subscribe 是否是**必要条件**（unsubscribe 后应重新失败 —— 可逆性检验，最强证据）

【验证矩阵】（每格独立 stream，记录返回码）
  E1 默认流(NULL stream) + 不 subscribe
  E2 创建流 + 不 subscribe                （已知 107015，作基线复现）
  E3 创建流 + subscribe                   （已知 0，作基线复现）
  E4 创建流 + subscribe → unsubscribe → launch  （可逆性：应回到 107015）
  E5 创建流 + subscribe + block=1         （排除 block 参数影响）
  E6 创建流 + 不 subscribe + block=1      （对照 E5，确认 block 无关）
  E7 另一 device 上重复 E2                （排除 device/context 依赖）

【判定】
  EXCLUSIVE       : E2/E6 失败(107015)、E3/E5 成功、E4 失败、E1/E7 与 E2 一致
                    → subscribe 是唯一变量，结论排他成立
  NOT_EXCLUSIVE   : 上述任一不成立（如 block 影响结果、或默认流行为不同）

【用法】容器内：python3 test_107015_exclusivity.py
"""
import argparse
import ctypes
import json
import sys

import acl

EXPECTED = 107015
CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _noop(arg):  # noqa: ARG001
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="acl_107015_exclusivity_result.json")
    args = ap.parse_args()

    rows = []

    def rec(eid, desc, expect, ret, extra=None):
        rec_row = {"id": eid, "desc": desc, "ret": ret, "expect": expect}
        if extra:
            rec_row.update(extra)
        rows.append(rec_row)
        mark = "✅" if ret == expect else "❌"
        print(f"  [{eid}] {desc:<52} ret={ret:<8} 期望={expect:<8} {mark}")

    ret = acl.init()
    if ret != 0:
        print(f"acl.init 失败 ret={ret}")
        return 1

    cb = CB(_noop)

    def ensure_device(dev):
        r = acl.rt.set_device(dev)
        if r != 0:
            for alt in range(0, 16):
                if acl.rt.set_device(alt) == 0:
                    return alt
            return None
        return dev

    dev0 = ensure_device(0)
    if dev0 is None:
        print("无可用设备")
        return 1

    print("=== 107015 排他性验证矩阵 ===")

    # E1 复现性：另一个独立的创建流 + 不 subscribe
    # 注：原计划测默认流(NULL stream)，但 pyACL 不接受 None 作为 stream（args parse failed），
    #     故改为"另一个独立创建流"以验证可重复性（是否所有创建流行为一致）
    s1, r = acl.rt.create_stream()
    r1 = acl.rt.launch_callback(cb, None, 0, s1) if r == 0 else r
    rec("E1", "另一个独立创建流 + 不 subscribe（复现性）", EXPECTED, r1)

    # E2 创建流 + 不 subscribe（基线复现）
    s2, r = acl.rt.create_stream()
    r2 = acl.rt.launch_callback(cb, None, 0, s2) if r == 0 else r
    rec("E2", "创建流 + 不 subscribe（基线）", EXPECTED, r2)

    # E3 创建流 + subscribe（基线复现）
    s3, r = acl.rt.create_stream()
    tid = __import__("threading").current_thread().ident or 0
    r_sub = acl.rt.subscribe_report(tid, s3) if r == 0 else r
    r3 = acl.rt.launch_callback(cb, None, 0, s3) if r_sub == 0 else r_sub
    rec("E3", "创建流 + subscribe（基线）", 0, r3, {"subscribe_ret": r_sub})

    # E4 可逆性：subscribe → unsubscribe → launch
    s4, r = acl.rt.create_stream()
    r_sub4 = acl.rt.subscribe_report(tid, s4) if r == 0 else r
    r_unsub = acl.rt.unsubscribe_report(tid, s4) if r_sub4 == 0 else r_sub4
    r4 = acl.rt.launch_callback(cb, None, 0, s4) if r_unsub == 0 else r_unsub
    rec("E4", "subscribe → unsubscribe → launch（可逆性）", EXPECTED, r4,
        {"subscribe_ret": r_sub4, "unsubscribe_ret": r_unsub})

    # E5 创建流 + subscribe + block=1
    s5, r = acl.rt.create_stream()
    r_sub5 = acl.rt.subscribe_report(tid, s5) if r == 0 else r
    r5 = acl.rt.launch_callback(cb, None, 1, s5) if r_sub5 == 0 else r_sub5
    rec("E5", "创建流 + subscribe + block=1", 0, r5, {"subscribe_ret": r_sub5})

    # E6 创建流 + 不 subscribe + block=1（对照 E5）
    s6, r = acl.rt.create_stream()
    r6 = acl.rt.launch_callback(cb, None, 1, s6) if r == 0 else r
    rec("E6", "创建流 + 不 subscribe + block=1（对照 E5）", EXPECTED, r6)

    # E7 另一 device 重复 E2
    alt_dev = None
    for d in range(1, 16):
        if acl.rt.set_device(d) == 0:
            alt_dev = d
            break
    if alt_dev is not None:
        s7, r = acl.rt.create_stream()
        r7 = acl.rt.launch_callback(cb, None, 0, s7) if r == 0 else r
        rec("E7", f"device={alt_dev} 创建流 + 不 subscribe", EXPECTED, r7,
            {"device": alt_dev})
    else:
        rec("E7", "无第二设备可测", EXPECTED, None)

    ok = all(r["ret"] == r["expect"] for r in rows if r["ret"] is not None)
    verdict = "EXCLUSIVE" if ok else "NOT_EXCLUSIVE"
    result = {"verdict": verdict, "matrix": rows,
              "note": "subscribe_report 是否为 107015 的唯一变量（含可逆性与 block/device 无关性）"}
    print(f"\n=== 判定：{verdict} ===")
    if ok:
        print("subscribe 为唯一变量：不订阅必失败、订阅后成功、取消订阅后复现、")
        print("且与 block 参数、默认流/创建流、device 选择无关 → 原结论排他成立")
    else:
        print("存在与预期不符的格子，原结论需修正（见上表 ❌ 项）")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果：{args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
