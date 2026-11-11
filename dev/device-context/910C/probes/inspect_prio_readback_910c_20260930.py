#!/usr/bin/env python3
"""910C：优先级参数「进没进去」的决定性核查（用 C API 回读优先级）。

依据：CANN 头文件 `acl_rt.h`
  · `aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)` —— 注释: value range 0~7
  · `aclrtStreamGetPriority(aclrtStream stream, uint32_t *priority)`        —— 可**回读**流优先级
  但 pyACL 只暴露了 `device_get_stream_priority_range`，**没有** `stream_get_priority` ⇒ 用 ctypes 直调 C API。

判据（先写死）：
  R1 能用 `create_stream_with_config(priority, flag)` 建流（pyACL 形式）
  R2 **回读**该流的 priority：若 == 传入值 ⇒ 参数**被接受并存下来**；若恒为 0 ⇒ 参数**被丢弃**
  R3 对照：priority=0 与 7 两条流回读值是否**不同**（不同 = 至少存下来了）
"""
import ctypes
import os
import time

import acl  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402

ns = torch.npu


def p(*a):
    print(*a, flush=True)


p("=== 前置：建立 ACL context ===")
torch.zeros(1, device="npu:0")
ns.set_device(0)
ns.synchronize()
p("  OK；range =", repr(acl.rt.device_get_stream_priority_range()))

# ── 找到 libascendcl 并声明两个 C 函数 ──
# ⚠️ 坑：工具链里同时有 **真实库** 与 **stub 库**（供编译期用）；dlopen 到 stub 会得到
#    rc=100039「stub library cannot be used for execution」。⇒ 从 **/proc/self/maps** 取
#    pyACL **已经加载的真实库**路径，再 CDLL 它。
lib = None
loaded = []
try:
    with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if "libascendcl.so" in ln or "libacl.so" in ln:
                path = ln.split()[-1]
                if path.startswith("/") and path not in loaded:
                    loaded.append(path)
except Exception as e:  # noqa: BLE001
    p("  读 maps 失败:", e)
p("  已加载的真实库（来自 /proc/self/maps）:", loaded[:4])
cands = loaded
for c in cands:
    try:
        lib = ctypes.CDLL(c)
        break
    except OSError:
        continue
if lib is None:
    # 退路：pyACL 自己已加载过 libacl，用 dlopen 找已加载句柄
    try:
        lib = ctypes.CDLL("libascendcl.so")
    except OSError as e:
        p("  ⛔ 加载失败:", e)
        raise SystemExit(0)

lib.aclrtStreamGetPriority.restype = ctypes.c_int
lib.aclrtStreamGetPriority.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
lib.aclrtCreateStreamWithConfig.restype = ctypes.c_int
lib.aclrtCreateStreamWithConfig.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint32]

FLAG_FAST_LAUNCH = 0  # ACL_STREAM_FAST_LAUNCH 枚举值；用 0 走默认也应是合法调用
res = {}

p("\n=== R1/R2 用 create_stream_with_config 建流并回读 priority ===")
for prio in (0, 3, 7):
    h = ctypes.c_void_p(0)
    rc = lib.aclrtCreateStreamWithConfig(ctypes.byref(h), ctypes.c_uint32(prio), ctypes.c_uint32(FLAG_FAST_LAUNCH))
    got = ctypes.c_uint32(0xFFFFFFFF)
    rc2 = lib.aclrtStreamGetPriority(ctypes.c_void_p(h.value), ctypes.byref(got))
    res[prio] = (rc, h.value, rc2, got.value)
    p(f"  传入 priority={prio}: create rc={rc} handle={h.value} | "
      f"回读 rc={rc2} priority={got.value}"
      + ("" if got.value == prio else f"  ⚠️ 与传入值不同（{got.value} != {prio}）"))

p("\n=== R2b 对照：torch.npu.Stream(priority=…) 建出来的流回读 priority ===")
for prio in (0, 7):
    s = ns.Stream(priority=prio)
    handle = getattr(s, "npu_stream", None)
    got = ctypes.c_uint32(0xFFFFFFFF)
    rc2 = lib.aclrtStreamGetPriority(ctypes.c_void_p(handle), ctypes.byref(got)) if handle else -1
    p(f"  torch Stream(priority={prio}): handle={handle} 回读 rc={rc2} priority={got.value}")

p("\n=== R3 判定 ===")
vals = {k: v[3] for k, v in res.items()}
distinct = len(set(vals.values())) > 1
accepted = all(vals[k] == k for k in vals)
p(f"  pyACL 三档回读值 = {vals}")
p(f"  · 参数被接受（回读 == 传入）: {accepted}")
p(f"  · 不同档位回读不同（至少存下来了）: {distinct}")
p("  ⇒ 若都是同一个值 ⇒ **优先级参数在本产品线上未生效**（与官方文档「训练产品上为预留参数」一致）")

p("\nBOUNDARY: 只读 + 稀疏建流；未改任何配置；结论只在本栈（CANN 9.0.0 / 910C / torch_npu 2.11.0）成立。")
