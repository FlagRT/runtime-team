#!/usr/bin/env python3
"""PPU 能力探测 · 第二轮（**只探测剩下的"能不能声明"决策项**）。

第一轮（`probe_ppu_stack_survey.py`）已确认接口形态；本轮的 6 项是**声明前必须实测**的：

  P. 流优先级**是否能真设置**（判据 = 能设置 **且** 回读一致 —— 不看构造参数回显）
  Q. 上下文生命周期：平台是否允许本层建第二个上下文（P800 是不允许）
  R. 是否存在**设备级重置原语**（决定 `recovery_real`）
  S. 图捕获是否真能跑（决定 `graph_capture`）
  T. 厂商错误码是否**透出到 Python**（决定 `error_map`）
  U. 选卡变量是哪个（`CUDA_VISIBLE_DEVICES` 是否生效）
  V. `torch.xpu` 是否也能看到设备（再确认走 B 而不是 C）

⚠️ 纪律：
  · **会改进程级状态的项（上下文创建/销毁）一律放子进程**，主进程不受影响
    （实测过：在进程内破坏设备上下文会让后续判据崩掉甚至段错误）。
  · 每项独立 try/except；取不到写 `UNAVAILABLE(...)`，**不猜、不补零**。
  · 判定"是否生效"一律用**独立回读**，不用构造参数回显（那是空转判据）。
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import textwrap

RESULT = []


def rec(k, v):
    RESULT.append((k, v))
    print(f"{k} = {v}", flush=True)


def probe(k, fn):
    try:
        rec(k, fn())
    except BaseException as e:                                   # noqa: BLE001
        rec(k, f"UNAVAILABLE({type(e).__name__}: {str(e)[:200]})")


def run_child(code: str, timeout: int = 120):
    """把一段代码放到**独立子进程**里跑，返回其 stdout（隔离进程级副作用）。"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(textwrap.dedent(code))
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return (p.stdout or "") + (("\n[stderr] " + p.stderr[-400:]) if p.returncode else "")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    import torch

    print("=" * 78)
    print("PPU 能力探测 · 第二轮（声明决策项）")
    print("=" * 78)

    # ── P. 流优先级：能设置 + 回读一致（真原语路径）────────────────────
    def _priority():
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuCtxGetStreamPriorityRange.restype = ctypes.c_int
        lib.cuCtxGetStreamPriorityRange.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        least, greatest = ctypes.c_int(0), ctypes.c_int(0)
        rc_rng = int(lib.cuCtxGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest)))
        out = {"range_rc": rc_rng, "least": least.value, "greatest": greatest.value}

        lib.cuStreamCreateWithPriority.restype = ctypes.c_int
        lib.cuStreamCreateWithPriority.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        lib.cuStreamGetPriority.restype = ctypes.c_int
        lib.cuStreamGetPriority.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.cuStreamDestroy_v2.restype = ctypes.c_int
        lib.cuStreamDestroy_v2.argtypes = [ctypes.c_void_p]

        # 对区间内每个取值做「建流 → 独立回读 → 销毁」
        seen = {}
        for prio in range(min(least.value, greatest.value), max(least.value, greatest.value) + 1):
            h = ctypes.c_void_p(0)
            rc_c = int(lib.cuStreamCreateWithPriority(ctypes.byref(h), ctypes.c_uint(2), ctypes.c_int(prio)))
            got = ctypes.c_int(-999)
            rc_g = int(lib.cuStreamGetPriority(h, ctypes.byref(got))) if rc_c == 0 else None
            seen[str(prio)] = {"create_rc": rc_c, "readback_rc": rc_g, "readback": got.value}
            if h.value:
                lib.cuStreamDestroy_v2(ctypes.c_void_p(h.value))
        # 再验一次：C API 建的流能否被 torch 包装并真跑算子（"这条流真的进了 torch 执行面"）
        h = ctypes.c_void_p(0)
        rc_c = int(lib.cuStreamCreateWithPriority(ctypes.byref(h), ctypes.c_uint(2), ctypes.c_int(greatest.value)))
        wrap = "NO"
        try:
            st = torch.cuda.ExternalStream(h.value, device=torch.cuda.current_device())
            with torch.cuda.stream(st):
                x = torch.ones(4, 4, device="cuda")
                y = float(x.sum().item())
            wrap = f"OK(sum={y})"
        except BaseException as e:                                # noqa: BLE001
            wrap = f"FAIL({type(e).__name__})"
        finally:
            if h.value:
                lib.cuStreamDestroy_v2(ctypes.c_void_p(h.value))
        out["per_value"] = seen
        out["external_stream_wrap"] = wrap
        # 构造参数回显 vs 真回读：对照（证明"读属性"这条路会空转）
        try:
            out["torch_ctor_attr_echo"] = torch.cuda.Stream(priority=greatest.value).priority
        except BaseException as e:                                # noqa: BLE001
            out["torch_ctor_attr_echo"] = f"UNAVAILABLE({type(e).__name__})"
        return json.dumps(out, ensure_ascii=False)

    probe("P1.priority_range_and_set_readback", _priority)

    # ── Q. 上下文生命周期（**子进程隔离**）──────────────────────────────
    probe("Q1.context_lifecycle_probe", lambda: run_child("""
        import ctypes, torch
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuCtxGetCurrent.restype = ctypes.c_int
        lib.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cuCtxCreate_v2.restype = ctypes.c_int
        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        lib.cuCtxDestroy_v2.restype = ctypes.c_int
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        cur = ctypes.c_void_p(0)
        rc_cur = int(lib.cuCtxGetCurrent(ctypes.byref(cur)))
        print("ctx_before_torch:", rc_cur, bool(cur.value))
        # 触碰设备（torch 会建立自己的上下文）
        a = torch.ones(2, 2, device="cuda"); print("torch_ok:", float(a.sum()))
        cur2 = ctypes.c_void_p(0); lib.cuCtxGetCurrent(ctypes.byref(cur2))
        print("ctx_after_torch:", bool(cur2.value))
        # 尝试建第二个上下文
        new = ctypes.c_void_p(0)
        rc_new = int(lib.cuCtxCreate_v2(ctypes.byref(new), ctypes.c_uint(0), ctypes.c_int(0)))
        print("cuCtxCreate_v2_rc:", rc_new, "handle:", bool(new.value))
        if new.value:
            rc_d = int(lib.cuCtxDestroy_v2(ctypes.c_void_p(new.value)))
            print("cuCtxDestroy_v2_rc:", rc_d)
        # 销毁后 torch 是否仍可用
        try:
            b = torch.ones(2, 2, device="cuda"); print("torch_after_ctx_ops:", float(b.sum()))
        except BaseException as e:
            print("torch_after_ctx_ops_FAIL:", type(e).__name__, str(e)[:120])
    """))

    # ── R. 设备级重置原语（决定 recovery_real；**子进程隔离**）──────────
    probe("R1.device_reset_primitives", lambda: run_child("""
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        cands = ["cuDevicePrimaryCtxReset_v2", "cuCtxResetPersistingL2Cache", "cuDeviceReset",
                 "cuDevicePrimaryCtxRelease_v2", "cuCtxResetPersistingL2Cache"]
        for c in cands:
            print(c, "has=", hasattr(lib, c))
        # torch 侧是否有设备级 reset（vs 仅内存统计类）
        import torch
        rs = [n for n in dir(torch.cuda) if n.startswith("reset")]
        print("torch_cuda_reset_like:", rs)
    """))

    # ── S. 图捕获是否真能跑（决定 graph_capture）────────────────────────
    probe("S1.graph_capture_real_run", lambda: run_child("""
        import torch
        try:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    x = torch.randn(64, 64, device="cuda"); y = x @ x
            torch.cuda.current_stream().wait_stream(s)
            with torch.cuda.graph(g):
                y2 = x @ x
            # 重放并对照
            x.copy_(torch.ones(64, 64, device="cuda"))
            g.replay(); torch.cuda.synchronize()
            ok = bool(torch.allclose(y2, torch.full_like(y2, 64.0)))
            print("GRAPH_CAPTURE:", "PASS" if ok else f"FAIL(value={float(y2[0,0])})")
        except BaseException as e:
            print("GRAPH_CAPTURE: FAIL(", type(e).__name__, str(e)[:160], ")")
    """))

    # ── T. 厂商错误码是否透出到 Python（决定 error_map）─────────────────
    probe("T1.vendor_error_code_surfacing", lambda: run_child("""
        import torch
        # 造一个必然失败的设备操作，看异常里有没有**数字码**
        cases = []
        try:
            torch.cuda.set_device(9999)
        except BaseException as e:
            cases.append(("set_device(9999)", type(e).__name__, str(e)[:200]))
        try:
            torch.zeros(4, 4, device="cuda:9999")
        except BaseException as e:
            cases.append(("zeros(cuda:9999)", type(e).__name__, str(e)[:200]))
        try:
            torch.cuda.mem_get_info(9999)
        except BaseException as e:
            cases.append(("mem_get_info(9999)", type(e).__name__, str(e)[:200]))
        import re
        for name, typ, msg in cases:
            nums = re.findall(r"\\b\\d{3,6}\\b", msg)
            print(f"{name}: {typ} | 数字码={nums} | {msg[:150]}")
    """))

    # ── U. 选卡变量（决定 DEV / *_VISIBLE_DEVICES 写法）─────────────────
    probe("U1.CUDA_VISIBLE_DEVICES_effect", lambda: run_child("""
        import os, subprocess, sys
        for val in ("0", "1", ""):
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = val
            code = "import torch;print('count=',torch.cuda.device_count())"
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            print(f"CUDA_VISIBLE_DEVICES={val!r} -> {p.stdout.strip() or p.stderr.strip()[-120:]}")
    """))

    # ── V. torch.xpu 是否也能看到设备（再确认 B 不是 C）─────────────────
    probe("V1.torch_xpu_device_count", lambda: (
        lambda: (str(torch.xpu.device_count()) if hasattr(torch, "xpu") else "NO_ATTR"))())

    print("=" * 78)
    unavail = [k for k, v in RESULT if isinstance(v, str) and v.startswith("UNAVAILABLE")]
    print(f"第二轮完成：{len(RESULT)} 项，取不到 {len(unavail)} 项 {unavail if unavail else ''}")
    print("=" * 78)


if __name__ == "__main__":
    main()
