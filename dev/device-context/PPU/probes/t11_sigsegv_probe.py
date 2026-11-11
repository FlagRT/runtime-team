#!/usr/bin/env python3
"""
t11_sigsegv_probe.py — 自装 SIGSEGV 取证处理器（不依赖 gdb；执行单 T11）
═══════════════════════════════════════════════════════════════════════════════

【为什么用这个办法】
  容器内**没有 gdb / eu-stack**（宿主有 gdb，但容器进程属 root、无免密 sudo ⇒ 不能从宿主 attach），
  也没有可用的 core dump 落点（`core_pattern` 指向宿主 `systemd-coredump`，需 root/权限组）。
  于是改用 **`sigaction(SA_SIGINFO)` 自装处理器**，在崩溃瞬间采集：
    ① `si_signo` / `si_code` / **`si_addr`**（故障内存地址）；
    ② 从 `ucontext_t` 里取 **`REG_RIP`**（故障指令地址）；
    ③ 用 `/proc/self/maps` 把 RIP 归属到 **具体 .so + 文件内偏移** —— 这就是"归属"的硬证据。
  再把 `.so 路径 + 偏移` 交给容器内已有的 `addr2line -f -C -e <so> <偏移>` 取符号名。

【纪律】
  · 本处理器只做三件事：读内存、`os.write(2,…)`、`os._exit(139)`；**不回调进 Python 解释器**
    （信号处理器内跑 Python 代码不安全；`os.write`/`os._exit` 是最小暴露面）。
  · 退出码固定 **139**，与"真崩溃"保持同一签名 ⇒ 与既有 rc 判据可比；
    并以 `T11_CRASH_CAPTURED` 标记区分"被采集"与"未被采集"。
  · ⚠️ 布局：x86_64 glibc 的 `ucontext_t` = `{uc_flags(8), uc_link(8), stack_t uc_stack(24), mcontext}`，
    `mcontext.gregs[23]` 里 **REG_RIP = 第 16 项**（R8..R15=0..7, RDI=8, RSI=9, RBP=10, RBX=11,
    RDX=12, RAX=13, RCX=14, RSP=15, RIP=16）⇒ RIP 位于 `uc + 40 + 16*8 = uc + 168`。
  · 通过 `sitecustomize.py` 注入（`PYTHONPATH=<本目录>`）⇒ **不改仓库代码**，载体保持原样。

【用法】
  PYTHONPATH=<本目录> python3 runtime/proto/proto_error_recovery_loop.py --backend ppu
"""
import ctypes
import os
import sys

_SA_SIGINFO = 0x00000004
_SA_RESTART = 0x10000000
_SIGSEGV = 11


class _Siginfo(ctypes.Structure):
    _fields_ = [
        ("si_signo", ctypes.c_int),
        ("si_errno", ctypes.c_int),
        ("si_code", ctypes.c_int),
        ("si_addr", ctypes.c_void_p),
    ]


class _Sigaction(ctypes.Structure):
    # glibc x86_64: handler(8) + sigset_t(128) + flags(4, 4 pad) + restorer(8)
    _fields_ = [
        ("sa_handler", ctypes.c_void_p),
        ("sa_mask", ctypes.c_ulonglong * 16),
        ("sa_flags", ctypes.c_int),
        ("sa_restorer", ctypes.c_void_p),
    ]


_HANDLER_T = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.POINTER(_Siginfo), ctypes.c_void_p)
_state = {"done": False, "handler": None}


def _maps_lookup(addr: int) -> str:
    """把地址归属到 `路径 + 文件内偏移`（读 `/proc/self/maps`，纯文本解析）。"""
    best = None
    try:
        with open("/proc/self/maps", "r") as fh:
            for line in fh:
                parts = line.split()
                if not parts or "-" not in parts[0]:
                    continue
                lo_s, hi_s = parts[0].split("-")
                lo, hi = int(lo_s, 16), int(hi_s, 16)
                if lo <= addr < hi:
                    name = parts[5] if len(parts) >= 6 else "[anon]"
                    off = int(parts[2], 16)
                    best = f"{name}+0x{addr - lo + off:x}（段内 0x{addr - lo:x}）路径={name}"
    except Exception:                                                    # noqa: BLE001
        pass
    return best or "（未在任何已映射段中）"


def _on_sigsegv(signo, si_ptr, uc_ptr):
    if _state["done"]:
        return
    _state["done"] = True
    try:
        si = si_ptr.contents if si_ptr else None
        addr = si.si_addr if si else None
        code = si.si_code if si else None
        rip = None
        if uc_ptr:
            base = int(uc_ptr)
            rip = int(ctypes.c_ulonglong.from_address(base + 168).value)
            rsp = int(ctypes.c_ulonglong.from_address(base + 40 + 15 * 8).value)
        else:
            rsp = 0
        out = [
            "",
            "===== T11_CRASH_CAPTURED =====",
            f"signal={signo} si_code={code} si_addr={addr if addr is None else hex(int(addr))}",
            f"RIP={hex(rip) if rip else '?'} RSP={hex(rsp)}",
            f"RIP 归属: {_maps_lookup(rip) if rip else '?'}",
            f"si_addr 归属: {_maps_lookup(int(addr)) if addr else '?'}",
            f"pid={os.getpid()}",
        ]
        os.write(2, ("\n".join(out) + "\n").encode("utf-8", "replace"))
    except Exception:                                                    # noqa: BLE001
        try:
            # ⚠️ 只能是 ASCII —— bytes 字面量里放非 ASCII 会让**整份模块**语法失败
            #    （实测踩过：sitecustomize 直接 SyntaxError ⇒ 处理器根本没装上，却静默继续跑）
            os.write(2, b"\nT11_CRASH_CAPTURED (handler internal error)\n")
        except Exception:                                                # noqa: BLE001
            pass
    os._exit(139)


def install() -> bool:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        h = _HANDLER_T(_on_sigsegv)
        _state["handler"] = h                      # 防被 GC 回收
        sa = _Sigaction()
        sa.sa_handler = ctypes.cast(h, ctypes.c_void_p)
        sa.sa_flags = _SA_SIGINFO | _SA_RESTART
        sa.sa_restorer = None
        rc = libc.sigaction(_SIGSEGV, ctypes.byref(sa), None)
        # ⭐ **回读校验**（本项目纪律：能设置 ≠ 已生效；必须"设了能读回一致"）——
        #    否则出现"处理器没装上、进程照样崩、却看起来一切正常"这种最贵的假象。
        old = _Sigaction()
        libc.sigaction(_SIGSEGV, None, ctypes.byref(old))
        ok = int(old.sa_handler or 0) == int(ctypes.cast(h, ctypes.c_void_p).value or 0)
        sys.stderr.write(
            f"[t11] SIGSEGV handler installed rc={rc} readback_match={ok} "
            f"installed=0x{int(ctypes.cast(h, ctypes.c_void_p).value or 0):x} "
            f"readback=0x{int(old.sa_handler or 0):x}\n")
        sys.stderr.flush()
        return rc == 0 and ok
    except Exception as e:                                               # noqa: BLE001
        sys.stderr.write(f"[t11] handler install failed: {e}\n")
        return False


_ok = install()
if not _ok:
    sys.stderr.write("[t11] !! handler NOT installed -> crash attribution unavailable\n")
    sys.stderr.flush()
