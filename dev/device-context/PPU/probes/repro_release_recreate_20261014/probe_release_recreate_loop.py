#!/usr/bin/env python3
"""PPU「释放后再建」误判复现器（T6 D2 失败根因）—— 模拟探针 run_group 的循环。

现象：新建的优先级流在**使用点**被判为「已被 release_stream() 释放」。
本复现器按 T6 探针的节奏（建 → 用 → 释放，多轮）循环，并在**命中瞬间** dump 内部状态：
  · 新流的 id / 类型 / 句柄
  · `_released_streams` 里同 id 条目的句柄、持有器类型、持有器解析结果是否**就是**新对象
  · `_owned_streams` 里是否也有同 id 条目

用法：docker exec <容器> bash -lc "cd <原型根> && DC_BACKEND=ppu CUDA_VISIBLE_DEVICES=1 \
        python3 -u <此脚本> [轮数]"
"""
import gc
import os
import sys

_ROOT = os.environ.get("DC_ROOT", "/workspace/runtime-team/dev/device-context/prototype")
sys.path.insert(0, _ROOT)

import runtime  # noqa: E402

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
PRIO = -3

runtime.use(os.environ.get("DC_BACKEND", "ppu"))
runtime.set_device(0)
bk = runtime.current()


def dump_hit(tag, native, exc):
    """命中瞬间的现场（不修状态，只读）。"""
    rid = id(native)
    rel = bk._reg("_released_streams")
    own = bk._reg("_owned_streams")
    print(f"\n[命中] {tag}: {type(exc).__name__}: {str(exc)[:90]}")
    print(f"  native: id={rid} type={type(native).__name__} handle={native.cuda_stream:#x}")
    e = rel.get(rid)
    if e is None:
        print("  _released_streams: 无同 id 条目  ← 与报错不符（说明键不是 id 或已被清）")
    else:
        h, holder = e[0], e[1]
        try:
            got = holder()
        except BaseException as ex:                       # noqa: BLE001
            got = f"<holder() 抛 {type(ex).__name__}>"
        print(f"  _released_streams[id]: 句柄={h:#x} 持有器类型={type(holder).__name__}")
        print(f"     holder() is native = {got is native}；holder() repr={repr(got)[:80]}")
    oe = own.get(rid)
    print(f"  _owned_streams: {'有条目（本层认为拥有）' if oe else '无条目'}")
    print(f"  _released_streams 表大小={len(rel)}  _owned_streams 表大小={len(own)}")


hits = 0
for i in range(1, ROUNDS + 1):
    s = runtime.create_stream(priority=PRIO)
    native = s._native_obj
    try:
        s.synchronize()                     # ← 使用点拦截在这里触发
        ok = True
    except BaseException as e:              # noqa: BLE001
        ok = False
        hits += 1
        dump_hit(f"round {i}", native, e)
        if hits >= 3:                       # 取三次现场即可
            break
    # 释放（模拟 run_group 的 finally）
    runtime.release_stream(s)
    if i % 10 == 0:
        gc.collect()                        # 模拟探针里大量的临时对象回收

print(f"\n=== 复现结果：{hits} 次命中 / {min(i, ROUNDS)} 轮 ===")
print("VERDICT:", "REPRODUCED" if hits else "NOT_REPRODUCED")
