#!/usr/bin/env python3
"""P800：流优先级是否真的可达（CUDA 兼容层回读核查）。

思路与 910C 对称：用 XPytorch 的 CUDA 兼容层（libcuda.so.1）
  · `cuCtxGetStreamPriorityRange(&least, &greatest)`  查范围
  · `cuStreamCreateWithPriority(&h, flags, priority)` 建流（带优先级）
  · `cuStreamGetPriority(h, &p)`                      回读
  · 对照：`torch.cuda.Stream(priority=…)` 建出来的流回读值
判据：
  K1 兼容层是否暴露这三个入口
  K2 `cuStreamCreateWithPriority` 能否建流、回读是否等于传入
  K3 `torch.cuda.Stream(priority=…)` 回读是否等于传入（⇒ 判断插件层有没有丢参数）
"""
import ctypes
import os

import torch  # noqa: E402

p = print

p("=== 前置 ===")
p("  torch", torch.__version__, "| cuda avail", torch.cuda.is_available(),
  "| visible", torch.cuda.device_count())
torch.cuda.set_device(0)
x = torch.arange(1, 65, dtype=torch.float32, device="cuda")
p("  matmul 校验值 =", float((x @ torch.tril(torch.ones(64, 64, device="cuda"))).sum().item()), "（期望 89440）")

# 找已加载的真实 libcuda（避开 stub）
loaded = []
with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
    for ln in f:
        if "libcuda.so" in ln or "libxpucuda" in ln:
            path = ln.split()[-1]
            if path.startswith("/") and path not in loaded:
                loaded.append(path)
p("  已加载:", loaded[:4])
lib = None
for c in loaded:
    try:
        lib = ctypes.CDLL(c)
        break
    except OSError:
        continue
if lib is None:
    p("  ⛔ 没找到可用的 cuda 兼容库")
    raise SystemExit(0)

p("\n=== K1 入口存在性 ===")
names = {}
for n in ("cuCtxGetStreamPriorityRange", "cuStreamCreateWithPriority", "cuStreamGetPriority"):
    try:
        getattr(lib, n)
        names[n] = True
    except AttributeError:
        names[n] = False
p("  ", names)

p("\n=== K1b 范围查询 ===")
least = ctypes.c_int(0)
greatest = ctypes.c_int(0)
try:
    rc = lib.cuCtxGetStreamPriorityRange(ctypes.byref(least), ctypes.byref(greatest))
    p(f"  cuCtxGetStreamPriorityRange rc={rc} least={least.value} greatest={greatest.value}")
except Exception as e:  # noqa: BLE001
    p("  调用失败:", type(e).__name__, e)

p("\n=== K2 cuStreamCreateWithPriority 建流 + 回读 ===")
try:
    lib.cuStreamCreateWithPriority.restype = ctypes.c_int
    lib.cuStreamCreateWithPriority.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
    lib.cuStreamGetPriority.restype = ctypes.c_int
    lib.cuStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    for prio in (0, -1, 1):
        h = ctypes.c_void_p(0)
        rc = lib.cuStreamCreateWithPriority(ctypes.byref(h), ctypes.c_uint(0), ctypes.c_int(prio))
        got = ctypes.c_int(0x7FFFFFFF)
        rc2 = lib.cuStreamGetPriority(ctypes.c_void_p(h.value), ctypes.byref(got)) if rc == 0 else -1
        p(f"  传入 {prio}: create rc={rc} handle={h.value} | 回读 rc={rc2} priority={got.value}")
except Exception as e:  # noqa: BLE001
    p("  K2 失败:", type(e).__name__, str(e)[:160])

p("\n=== K3 torch.cuda.Stream(priority=…) 回读 ===")
try:
    for prio in (0, -1):
        try:
            s = torch.cuda.Stream(priority=prio)
        except Exception as e:  # noqa: BLE001
            p(f"  torch Stream(priority={prio}) 建流失败: {type(e).__name__}: {str(e)[:110]}")
            continue
        h = getattr(s, "cuda_stream", None)
        got = ctypes.c_int(0x7FFFFFFF)
        rc2 = lib.cuStreamGetPriority(ctypes.c_void_p(h), ctypes.byref(got)) if h else -1
        p(f"  torch Stream(priority={prio}): handle={h} 回读 rc={rc2} priority={got.value}")
except Exception as e:  # noqa: BLE001
    p("  K3 失败:", type(e).__name__, str(e)[:160])

p("\nBOUNDARY: 只读 + 稀疏建流；结论只在 P800 / XPU-RT 5.0.21 / torch 2.9.0+cu129 成立。")
