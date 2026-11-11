#!/usr/bin/env python3
"""PPU：释放后流对象**为何仍存活**、以及 id 复用如何发生（T6 失败根因第二段）。

前置事实（已测）：
  · 单发：create(priority) → weakref → del + gc ⇒ **weakref 未置死**（对象仍活）
  · 循环：多轮建/释放后，`_released_streams` 里的 weakref 能解析出**新对象**
    ⇒ 两种可能：(a) 有东西强持有旧对象（则 id 不可能复用）或 (b) 对象确实死亡后地址被复用。
本脚本用 `gc.get_referrers` 直接问「谁持有它」，并逐轮记录 id 序列与两表状态。

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
bk = runtime.current()
PRIO = -3


def brief(r):
    try:
        s = repr(r)
    except BaseException:                                     # noqa: BLE001
        s = "<repr 失败>"
    return f"{type(r).__name__}: {s[:70]}"


print("=== 1) 释放后对象是否仍被持有（gc.get_referrers）===")
s = runtime.create_stream(priority=PRIO)
n = s._native_obj
w = weakref.ref(n)
print(f"  native id={id(n)} handle={n.cuda_stream:#x}")
runtime.release_stream(s)
del s, n
gc.collect()
o = w()
print(f"  release + del + gc 后：w() is None = {o is None}")
if o is not None:
    print(f"  sys.getrefcount(o) = {sys.getrefcount(o)}  （含本行临时引用）")
    refs = gc.get_referrers(o)
    print(f"  持有者 {len(refs)} 个：")
    for r in refs[:8]:
        print("    -", brief(r))
        if isinstance(r, dict):
            keys = [k for k, v in r.items() if v is o]
            print(f"        作为值出现在 dict 的键 {keys}（dict 大小 {len(r)}）")

print()
print("=== 2) 循环：id 序列与两表状态（不释放，观察 id 是否复用）===")
seen = {}
for i in range(1, 9):
    si = runtime.create_stream(priority=PRIO)
    ni = si._native_obj
    nid = id(ni)
    dup = seen.get(nid)
    seen[nid] = seen.get(nid, 0) + 1
    rel = bk._reg("_released_streams")
    ent = rel.get(nid)
    holder_state = "无条目"
    if ent is not None:
        got = ent[1]()
        holder_state = f"条目在册，holder() is native = {got is ni}"
    print(f"  round {i}: id={nid} handle={ni.cuda_stream:#x} "
          f"{'（该 id 第 %d 次出现）' % seen[nid] if dup else ''}｜{holder_state}")
    try:
        si.synchronize()
        use = "✅ 可用"
    except BaseException as e:                                # noqa: BLE001
        use = f"❌ {type(e).__name__}"
    print(f"            使用点：{use}")
    runtime.release_stream(si)
    del si, ni
    if i % 3 == 0:
        gc.collect()

print()
print("=== 3) 两表大小与持有器类型直方图 ===")
rel = bk._reg("_released_streams")
own = bk._reg("_owned_streams")
from collections import Counter
print("  _released_streams:", len(rel), "→", Counter(type(v[1]).__name__ for v in rel.values()))
print("  _owned_streams  :", len(own), "→", Counter(type(v[1]).__name__ for v in own.values()))
