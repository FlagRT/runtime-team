#!/usr/bin/env python3
"""PPU 能力探测 · 第三轮（补第二轮的两处方法学失误 + 四项待决）。

第二轮踩到的**方法学问题（已修正）**：
  ❗ **调厂商 C API 之前必须先"触碰设备"** —— 第二轮 P1 第一次直接 `ctypes` 调
     `cuCtxGetStreamPriorityRange`，拿到 `rc=3`（CUDA_ERROR_NOT_INITIALIZED）
     ⇒ 后续建流全失败。这正是《接入手册》坑 18 的同一族
     （"取设备命名空间 / 拼厂商专有字符串前必须先触碰设备"）。
     ⇒ 本轮每个子进程代码块**开头统一先跑一次设备张量**，再调 C API。

本轮要定的四件事：
  P2. 流优先级**真路径**是否「能设置 + 回读一致」（**用独立回读，不看构造参数回显**）
  Q2. 上下文：能否"建多个 + 切换 + 只销毁自己的"（`context_lifecycle` 的判据）
  R2. `cuDevicePrimaryCtxReset_v2` 是否可作为**设备级重建原语**（`recovery_real`）
  T2. 真实设备侧错误是否带**数字码**（`error_map` 的最后确认）

⚠️ 会改进程级状态的项（Q2 / R2）一律**子进程隔离**。

⚠️ 写代码时的坑（2026-10-10 踩到两次）：子进程代码块必须**只对块本身去缩进**。
   写成 `textwrap.dedent(HEAD + block)` 时 HEAD 已在 0 缩进 ⇒ 共同前缀 = 0
   ⇒ block 不被去缩进 ⇒ 子进程 `IndentationError`，而父进程只看到**空 stdout**，
   **极易误判成"平台不支持"**。本文件做法：块自带首部两行"触碰设备"，整体一起 dedent。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

RESULT = []

#: 子进程代码块统一首部：**先触碰设备**（否则 C API 拿到 rc=3 未初始化）
HEAD = """\
import ctypes, torch
_t = torch.ones(2, 2, device="cuda")
print("touch_device:", float(_t.sum()))
lib = ctypes.CDLL("libcuda.so.1")
"""


def rec(k, v):
    RESULT.append((k, v))
    print(f"{k} = {v}", flush=True)


def probe(k, fn):
    try:
        rec(k, fn())
    except BaseException as e:                                   # noqa: BLE001
        rec(k, f"UNAVAILABLE({type(e).__name__}: {str(e)[:200]})")


def run_child(block: str, timeout: int = 240):
    """把代码块（自带 HEAD）放进**独立子进程**跑，返回 stdout（含失败时 stderr 尾部）。"""
    src = HEAD + textwrap.dedent(block)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        out = p.stdout or ""
        if p.returncode != 0:
            out += f"\n[rc={p.returncode}] " + (p.stderr or "")[-500:]
        return out
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


P2_CODE = """
lib.cuCtxGetStreamPriorityRange.restype = ctypes.c_int
lib.cuCtxGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
least, greatest = ctypes.c_int(0), ctypes.c_int(0)
rc = int(lib.cuCtxGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)))
print("range_rc:", rc, "least:", least.value, "greatest:", greatest.value)

lib.cuStreamCreateWithPriority.restype = ctypes.c_int
lib.cuStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
lib.cuStreamGetPriority.restype = ctypes.c_int
lib.cuStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.cuStreamDestroy_v2.restype = ctypes.c_int
lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]

lo, hi = min(least.value, greatest.value), max(least.value, greatest.value)
for prio in range(lo, hi + 1):
    h = ctypes.c_void_p(0)
    rc_c = int(lib.cuStreamCreateWithPriority(ctypes.byref(h), ctypes.c_uint(2), ctypes.c_int(prio)))
    got = ctypes.c_int(-999); rc_g = None
    if rc_c == 0 and h.value:
        rc_g = int(lib.cuStreamGetPriority(h, ctypes.byref(got)))
    print("SET prio=%d create_rc=%d readback_rc=%s readback=%d" % (prio, rc_c, rc_g, got.value))
    if h.value:
        lib.cuStreamDestroy_v2(ctypes.c_void_p(h.value))

h = ctypes.c_void_p(0)
rc_c = int(lib.cuStreamCreateWithPriority(ctypes.byref(h), ctypes.c_uint(2), ctypes.c_int(greatest.value)))
try:
    st = torch.cuda.ExternalStream(h.value, device=torch.cuda.current_device())
    with torch.cuda.stream(st):
        s = float(torch.ones(4, 4, device="cuda").sum().item())
    print("external_stream_wrap: OK sum=", s)
except BaseException as e:
    print("external_stream_wrap: FAIL(", type(e).__name__, str(e)[:120], ")")
finally:
    if h.value:
        lib.cuStreamDestroy_v2(ctypes.c_void_p(h.value))

ts = torch.cuda.Stream()
try:
    raw = ts.cuda_stream
    got = ctypes.c_int(-999)
    rc_g = int(lib.cuStreamGetPriority(ctypes.c_void_p(int(raw)), ctypes.byref(got)))
    print("torch_stream_capi_readback: rc=%d value=%d attr=%s" % (rc_g, got.value, getattr(ts, "priority", "NA")))
except BaseException as e:
    print("torch_stream_capi_readback_FAIL:", type(e).__name__, str(e)[:120])
"""

Q2_CODE = """
lib.cuCtxGetCurrent.restype = ctypes.c_int
lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
lib.cuCtxCreate_v2.restype = ctypes.c_int
lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
lib.cuCtxDestroy_v2.restype = ctypes.c_int
lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
lib.cuCtxSetCurrent.restype = ctypes.c_int
lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]

def cur():
    c = ctypes.c_void_p(0); lib.cuCtxGetCurrent(ctypes.byref(c)); return c.value

origin = cur(); print("origin_ctx_nonnull:", bool(origin))
a = ctypes.c_void_p(0); rc_a = int(lib.cuCtxCreate_v2(ctypes.byref(a), ctypes.c_uint(0), ctypes.c_int(0)))
print("create_A_rc:", rc_a, "cur_is_A:", cur() == a.value)
b = ctypes.c_void_p(0); rc_b = int(lib.cuCtxCreate_v2(ctypes.byref(b), ctypes.c_uint(0), ctypes.c_int(0)))
print("create_B_rc:", rc_b, "B_ne_A:", a.value != b.value, "cur_is_B:", cur() == b.value)
rc_set = int(lib.cuCtxSetCurrent(ctypes.c_void_p(a.value)))
print("set_current_A_rc:", rc_set, "cur_is_A:", cur() == a.value)
rc_d = int(lib.cuCtxDestroy_v2(ctypes.c_void_p(a.value)))
print("destroy_A_rc:", rc_d)
rc_set2 = int(lib.cuCtxSetCurrent(ctypes.c_void_p(b.value)))
print("set_current_B_rc:", rc_set2)
try:
    print("torch_under_B:", float(torch.ones(2, 2, device="cuda").sum()))
except BaseException as e:
    print("torch_under_B_FAIL:", type(e).__name__, str(e)[:140])
print("destroy_B_rc:", int(lib.cuCtxDestroy_v2(ctypes.c_void_p(b.value))))
if origin:
    lib.cuCtxSetCurrent(ctypes.c_void_p(origin))
try:
    print("torch_after_restore:", float(torch.ones(2, 2, device="cuda").sum()))
except BaseException as e:
    print("torch_after_restore_FAIL:", type(e).__name__, str(e)[:140])
"""

R2_CODE = """
lib.cuDevicePrimaryCtxReset_v2.restype = ctypes.c_int
lib.cuDevicePrimaryCtxReset_v2.argtypes = [ctypes.c_int]
print("before_reset_torch_ok:", float(torch.ones(2, 2, device="cuda").sum()))
print("cuDevicePrimaryCtxReset_v2_rc:", int(lib.cuDevicePrimaryCtxReset_v2(ctypes.c_int(0))))
try:
    torch.cuda.synchronize(); print("sync_after_reset: OK")
except BaseException as e:
    print("sync_after_reset_FAIL:", type(e).__name__, str(e)[:120])
try:
    print("reset_then_plain:", float(torch.ones(2, 2, device="cuda").sum()))
except BaseException as e:
    print("reset_then_plain_FAIL:", type(e).__name__, str(e)[:140])
try:
    torch.cuda.set_device(0)
    print("reset_then_setdevice:", float(torch.ones(2, 2, device="cuda").sum()))
except BaseException as e:
    print("reset_then_setdevice_FAIL:", type(e).__name__, str(e)[:140])
"""

T2_CODE = """
import re
torch.ones(2, 2, device="cuda")
cases = []
try:
    torch.empty(int(200 * 1024**3), dtype=torch.uint8, device="cuda")
except BaseException as e:
    cases.append(("OOM", type(e).__name__, str(e)[:220]))
try:
    torch.zeros(1, device="cuda:7")
except BaseException as e:
    cases.append(("bad_ordinal", type(e).__name__, str(e)[:220]))
try:
    x = torch.ones(2, 2, device="cuda"); torch.cuda.synchronize()
except BaseException as e:
    cases.append(("misc", type(e).__name__, str(e)[:220]))
for name, typ, msg in cases:
    nums = re.findall(r"(?:error|code|rc)[^0-9]{0,6}([0-9]{3,6})", msg, re.I)
    print(name + ": " + typ + " | vendor_code_candidates=" + str(nums) + " | " + msg[:170])
if not cases:
    print("no_error_case_triggered")
"""


def main():
    print("=" * 78)
    print("PPU 能力探测 · 第三轮（先触碰设备，再调 C API）")
    print("=" * 78)
    probe("P2.priority_range_and_set_readback", lambda: run_child(P2_CODE))
    probe("Q2.context_multi_and_set", lambda: run_child(Q2_CODE))
    probe("R2.primary_ctx_reset_viability", lambda: run_child(R2_CODE))
    probe("T2.device_side_error_codes", lambda: run_child(T2_CODE))
    print("=" * 78)
    unavail = [k for k, v in RESULT if isinstance(v, str) and v.startswith("UNAVAILABLE")]
    print(f"第三轮完成：{len(RESULT)} 项，取不到 {len(unavail)} 项 {unavail if unavail else ''}")
    print("=" * 78)


if __name__ == "__main__":
    main()
