"""最小实验：PyTorch 的流对象被回收后，其 weakref 是否被正确置为 dead？

背景（本轮已确证的现场）：
  `_owned_streams[P]` 的条目由 register 为 **ExternalStream** 创建（同进程对齐、id 一致），
  但当 L1 的默认流 **Stream** 落在同一地址 P 时，该条目的 `weakref()` **返回了这个新 Stream**
  （repr 显示 `to 'Stream' at 0x…`，非 dead）⇒ 归属判定误判为"本层拥有" ⇒ 越权销毁。

若本实验显示「对象回收后 weakref 仍返回对象」⇒ 根因即 **厂商 C++ 类型的 weakref 未随回收清理（悬垂）**，
与「id 复用 + 弱引用无法区分同一性」是同一问题的**真正机制**。
"""
import gc
import weakref

import torch

print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)

for label, maker in (("torch.cuda.Stream()", lambda: torch.cuda.Stream()),):
    s = maker()
    w = weakref.ref(s)
    addr = id(s)
    print(f"\n=== {label} ===", flush=True)
    print(f"  type={type(s).__name__} id={hex(addr)}", flush=True)
    print(f"  回收前：w() 存活 = {w() is not None}  w() is s = {w() is s}", flush=True)
    del s
    gc.collect()
    got = w()
    print(f"  回收后（del + gc.collect）：w() = {got!r}", flush=True)
    print(f"  ⇒ 是否正确置死（应为 None）：{'✅ 是' if got is None else '❌ 否 —— weakref 悬垂！'}", flush=True)
    if got is not None:
        print(f"     被返回对象的类型 = {type(got).__name__}（id={hex(id(got))}）", flush=True)

    # 再分配，看是否新对象落在同一地址、且 w() 会"指向"它
    for i in range(200):
        t = torch.cuda.Stream()
        if id(t) == addr:
            print(f"  第 {i+1} 个新流落在同一地址 {hex(addr)}："
                  f"此时 w() 类型 = {type(w()).__name__ if w() is not None else None}, "
                  f"w() is t = {w() is t}", flush=True)
            break
    else:
        print("  （200 次分配内未出现同地址复用）", flush=True)

# 对照：普通 Python 类实例（CPython 会正确清理 weakref）
class Foo:
    pass


f = Foo()
wf = weakref.ref(f)
del f
gc.collect()
print(f"\n=== 对照：普通类实例 ===\n  回收后 wf() = {wf()!r} "
      f"⇒ {'✅ 正确置死' if wf() is None else '❌ 未置死'}", flush=True)
