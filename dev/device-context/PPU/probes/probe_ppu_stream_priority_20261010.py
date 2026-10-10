#!/usr/bin/env python3
"""PPU 能力探测 · 第四轮：钉死**流优先级**的真实形态（`stream_priority_control/readback` 的判据）。

第三轮在这一项上留下两个必须回答的问题：
  1. `cuStreamCreateWithPriority` 对 `-3..0` 全部返回 `create_rc=1`（INVALID_VALUE）
     ⇒ **声明区间 `(0, -3)` 与实际可建流取值域不一致**？还是我传的 flag / 参数有问题？
  2. `torch.cuda.Stream(priority=…)` 这条路能不能真把参数送进设备？
     —— 判据只能是「**建流后用独立 C API 回读**，且回读 == 请求」
     （⚠️ 绝不能读 `.priority` 属性：那是**构造参数回显**，MLU590 已实测为空转判据）。

做法（全部在**子进程**里，互不污染）：
  · A. C API 扫描：flag ∈ {0, 1, 2, 4} × priority ∈ [-8, 8]，记录 create_rc 与回读；
  · B. torch 路径：`torch.cuda.Stream(priority=p)` for p ∈ 区间内取值，**用 C API 回读**；
  · C. 对照：不指定优先级时的回读值（基线）。

⚠️ `c_void_p(0).value` 是 `None` 而非 0 —— 判空必须用 `if h.value is not None`，
   否则会把「建流失败」误判成「句柄为 0」再抛 TypeError，把真实原因盖掉。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

RESULT = []

HEAD = """\
import ctypes, torch
_t = torch.ones(2, 2, device="cuda")
lib = ctypes.CDLL("libcuda.so.1")
lib.cuStreamCreateWithPriority.restype = ctypes.c_int
lib.cuStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
lib.cuStreamGetPriority.restype = ctypes.c_int
lib.cuStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
lib.cuStreamDestroy_v2.restype = ctypes.c_int
lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]
lib.cuCtxGetStreamPriorityRange.restype = ctypes.c_int
lib.cuCtxGetStreamPriorityRange.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
"""

SCAN_CODE = """
least, greatest = ctypes.c_int(0), ctypes.c_int(0)
print("range_rc=%d least=%d greatest=%d" % (
    int(lib.cuCtxGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))),
    least.value, greatest.value))

def create(flag, prio):
    h = ctypes.c_void_p(0)
    rc = int(lib.cuStreamCreateWithPriority(ctypes.byref(h), ctypes.c_uint(flag), ctypes.c_int(prio)))
    got, rc_g = None, None
    if rc == 0 and h.value is not None:
        g = ctypes.c_int(-999)
        rc_g = int(lib.cuStreamGetPriority(h, ctypes.byref(g)))
        got = g.value
    if h.value is not None:
        lib.cuStreamDestroy_v2(h)
    return rc, rc_g, got

for flag in (0, 1, 2, 4):
    row = []
    for prio in range(-8, 9):
        rc, rc_g, got = create(flag, prio)
        row.append("%d:%s" % (prio, ("rc%d/get%s" % (rc, got)) if rc == 0 else ("rc%d" % rc)))
    print("FLAG=%d  " % flag + "  ".join(row))
"""

TORCH_CODE = """
# torch 路径：建流后用 **C API 独立回读**（不看 .priority 属性）
def capi_read(stream):
    try:
        g = ctypes.c_int(-999)
        rc = int(lib.cuStreamGetPriority(ctypes.c_void_p(int(stream.cuda_stream)), ctypes.byref(g)))
        return ("rc%d" % rc, g.value)
    except BaseException as e:
        return ("ERR", type(e).__name__)

base = torch.cuda.Stream()
print("BASELINE(no priority) capi_read:", capi_read(base), "attr:", getattr(base, "priority", "NA"))

for p in (-3, -2, -1, 0):
    try:
        st = torch.cuda.Stream(priority=p)
        print("torch Stream(priority=%d) -> capi_read=%s attr=%s" % (p, capi_read(st), getattr(st, "priority", "NA")))
    except BaseException as e:
        print("torch Stream(priority=%d) FAIL %s: %s" % (p, type(e).__name__, str(e)[:120]))

# 越界取值（看是否静默夹取 or 报错）
for p in (-4, 1, 7):
    try:
        st = torch.cuda.Stream(priority=p)
        print("torch Stream(priority=%d) OUT-OF-SPEC -> capi_read=%s" % (p, capi_read(st)))
    except BaseException as e:
        print("torch Stream(priority=%d) OUT-OF-SPEC FAIL %s: %s" % (p, type(e).__name__, str(e)[:120]))

# 真跑一次算子，确认带优先级的流确实能执行（不是"只看句柄"）
try:
    st = torch.cuda.Stream(priority=-3)
    with torch.cuda.stream(st):
        v = float((torch.ones(8, 8, device="cuda") @ torch.ones(8, 8, device="cuda")).sum().item())
    torch.cuda.synchronize()
    print("stream(priority=-3) real_op_sum:", v)
except BaseException as e:
    print("stream(priority=-3) real_op FAIL:", type(e).__name__, str(e)[:140])
"""


def main():
    print("=" * 78)
    print("PPU 能力探测 · 第四轮：流优先级真实取值域")
    print("=" * 78)
    for name, code in (("priority_scan", SCAN_CODE), ("torch_priority_path", TORCH_CODE)):
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(HEAD + textwrap.dedent(code))
            path = f.name
        try:
            p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=300)
            out = (p.stdout or "") + (("\n[rc=%d] " % p.returncode) + (p.stderr or "")[-400:] if p.returncode else "")
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        RESULT.append((name, out))
        print(f"---- {name}")
        print(out.rstrip())
    print("=" * 78)


if __name__ == "__main__":
    main()
