#!/usr/bin/env python3
"""910C：① 用 pyACL 直连找**流配额精确上限**；② 看 torch 建流会不会真的耗配额。

依据（官方文档）：
  · Atlas 训练系列：硬件最多 2048 个 Stream；`N = 2048 - 默认Stream - 内部同步Stream`
  · `aclrtGetStreamAvailableNum` = 还能显式创建多少个
实测上一轮：pyACL 建流 1:1 递减 available_num；torch 建 1000 条则**完全不变**。
本脚本把两边都推到**失败点**，拿到精确数字与错误码。
"""
import ctypes
import os
import re
import subprocess
import time

import acl  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402

MODE = os.environ.get("MODE", "pyacl")          # pyacl | torch
CAP = int(os.environ.get("CAP", "4000"))
ns = torch.npu


def p(*a):
    print(*a, flush=True)


p(f"=== MODE={MODE} CAP={CAP} ===")
# ⚠️ 必须先让 torch 建立 ACL context，否则直连 pyACL 会 rc=107002（context is null）
torch.zeros(1, device="npu:0")
ns.set_device(0)
ns.synchronize()
p("  [前置] torch 已触碰设备 ⇒ ACL context 已建立")
# 先看 _C.so 里用的是哪个建流 API（判断 torch 是否走 WithConfig/priority）
try:
    import torch_npu
    so = os.path.join(os.path.dirname(torch_npu.__file__), "_C.cpython-312-aarch64-linux-gnu.so")
    if os.path.exists(so):
        out = subprocess.run(["strings", so], capture_output=True, text=True, timeout=120).stdout
        hits = sorted({m for m in re.findall(r"aclrt[A-Za-z]*[Ss]tream[A-Za-z]*", out)})
        p("  _C.so 里出现的 aclrt*Stream* 符号：", hits[:25])
except Exception as e:  # noqa: BLE001
    p("  .so 扫描失败:", type(e).__name__, e)

p(f"  起始 available_num = {acl.rt.get_stream_available_num()!r}")

if MODE == "pyacl":
    kept = []
    t0 = time.monotonic()
    err = None
    try:
        for i in range(CAP):
            r = acl.rt.create_stream()
            # pyACL 返回 (stream, ret)
            if isinstance(r, tuple):
                st, rc = r
                if rc != 0:
                    err = f"rc={rc}（第 {i + 1} 条）"
                    break
                kept.append(st)
            else:
                kept.append(r)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}（第 {len(kept) + 1} 条）"
    dt = time.monotonic() - t0
    p(f"  pyACL 建流：成功 {len(kept)} 条，耗时 {dt:.2f}s，首次失败={err}")
    p(f"  建完后 available_num = {acl.rt.get_stream_available_num()!r}")
    # 关键判据：上限 = 成功条数 + 起始可用数
    try:
        _pre = None
        p("  ⇒ 上限 ≈ 起始 available_num + 成功条数（应与文档的 2048 − 默认/内部流 对得上）")
    except Exception:  # noqa: BLE001
        pass
else:
    kept = []
    t0 = time.monotonic()
    err = None
    try:
        for i in range(CAP):
            kept.append(ns.Stream())
            if (i + 1) % 500 == 0:
                p(f"    torch 建流 {i + 1:>5} 条  available_num={acl.rt.get_stream_available_num()!r}")
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {str(e)[:200]}（第 {len(kept) + 1} 条）"
    dt = time.monotonic() - t0
    p(f"  torch 建流：成功 {len(kept)} 条，耗时 {dt:.2f}s，首次失败={err}")
    p(f"  建完后 available_num = {acl.rt.get_stream_available_num()!r}")
    # 再**使用**其中若干条，看是否这时才真正占配额
    try:
        a = torch.full((4, 4), 2.0, device="npu:0")
        used = 0
        for s in kept[:300]:
            with ns.stream(s):
                _ = a.sum().item()
            used += 1
        ns.synchronize()
        p(f"  真的在 {used} 条 torch 流上跑了计算后  available_num={acl.rt.get_stream_available_num()!r}")
    except Exception as e:  # noqa: BLE001
        p("  使用流失败:", type(e).__name__, str(e)[:160])

p("\nBOUNDARY: 只读+建流；未调任何 set_* 配置接口；结论只在本栈（torch_npu 2.11.0 / CANN 9.0.0 / 910C 单卡）成立。")
