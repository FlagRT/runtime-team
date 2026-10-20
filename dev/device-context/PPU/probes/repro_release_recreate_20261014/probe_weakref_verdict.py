#!/usr/bin/env python3
"""决定性微实验：PPU 的 torch 流对象，其 weakref 在对象回收后是否**真的置死**？

背景：T6 D2 探针与循环复现器都出现「新建的流被判为已释放」，现场显示
`_released_streams` 中**为旧对象创建的 weakref** 解析出了**新对象**（地址复用）。
10-10 的单发最小实验曾显示 weakref 正确置死（当时否证了「悬垂」），故需给出
**能区分两种可能**的实验：

  A 场景（单发）：建 → 取 weakref → 删 → gc → 看 w()
  B 场景（地址复用）：… 再建一条流，看 w() 是否**变成**新对象
  C 场景（循环 + 周期性 gc）：模拟探针节奏，检查是否出现 w() 跟随地址

判读：
  · A 返回 None 且 B 仍 None  ⇒ 弱引用语义正常（则"悬垂"不成立，需另找机制）
  · A 返回 None 但 B 返回新对象 ⇒ **悬垂**（weakref 未随对象死亡失效）
用法：DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=1 python3 -u <此脚本>
"""
import gc
import os
import sys
import weakref

_ROOT = os.environ.get("DC_ROOT", "/workspace/runtime-team/dev/device-context/prototype")
sys.path.insert(0, _ROOT)

import runtime  # noqa: E402

runtime.use(os.environ.get("DC_BACKEND", "ppu"))
runtime.set_device(0)
PRIO = -3

print("=== A 场景：单发（建 → weakref → 删 → gc）===")
s = runtime.create_stream(priority=PRIO)
n = s._native_obj
w = weakref.ref(n)
print(f"  native id={id(n)} handle={n.cuda_stream:#x}；w() is native = {w() is n}")
del s, n
gc.collect()
print(f"  del + gc 后 w() = {w()!r}   ⇒ {'✅ 已置死' if w() is None else '❌ 未置死'}")

print("=== B 场景：再建一条，看同一 weakref 是否跟随地址 ===")
s2 = runtime.create_stream(priority=PRIO)
n2 = s2._native_obj
print(f"  new native id={id(n2)} handle={n2.cuda_stream:#x}")
got = w()
print(f"  w() = {got!r}")
print(f"  ▶ w() is n2 = {got is n2}   ⇒ {'❌ 悬垂（w 跟到了新对象）' if got is n2 else '✅ 未跟随'}")

print("=== C 场景：循环建/释放 + 周期 gc，统计弱引用失效率 ===")
bad = 0
kept = []
for i in range(30):
    si = runtime.create_stream(priority=PRIO)
    ni = si._native_obj
    wi = weakref.ref(ni)
    runtime.release_stream(si)
    del si, ni
    if i % 5 == 4:
        gc.collect()
    if wi() is not None:
        bad += 1
        if bad <= 3:
            cur = wi()
            print(f"  [第 {i} 轮] weakref 未置死：w() repr={repr(cur)[:70]} type={type(cur).__name__}")
    kept.append(wi)          # 故意持有 weakref（不持有对象）
print(f"  ⇒ 30 轮中 weakref 未置死 = {bad}")

print("=== 收尾 ===")
try:
    runtime.release_stream(s2)
except BaseException as e:                                    # noqa: BLE001
    print("  release(s2):", type(e).__name__)
print("VERDICT:",
      "WEAKREF_DANGLING" if (got is n2 or bad) else "WEAKREF_CORRECT")
