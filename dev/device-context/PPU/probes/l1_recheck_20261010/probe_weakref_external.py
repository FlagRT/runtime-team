"""关键补充实验：`torch.cuda.ExternalStream` 的 weakref 在回收后是否被置死？

上一实验只测了 `torch.cuda.Stream()`（结果：正确置死）。
但 `_register_owned_stream` 登记的是 **`torch.cuda.ExternalStream(handle, device=…)`**
（PPU backend `_create_stream_raw(priority=<int>)`）⇒ 必须单独测它的 weakref 行为。

若 ExternalStream 的 weakref 在对象回收后**仍返回一个对象**（而非 None）⇒
现场观测（`holder()` 返回新 Stream）即由此而来，根因定案。
"""
import gc
import weakref

import torch

print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)

base = torch.cuda.Stream()                       # 借一条真实流，取其设备侧句柄
handle = base.cuda_stream
print(f"[准备] 借用的 cuda_stream 句柄 = {hex(handle)}", flush=True)

print("\n=== torch.cuda.ExternalStream(handle) ===", flush=True)
es = torch.cuda.ExternalStream(handle, device=torch.cuda.current_device())
w = weakref.ref(es)
addr = id(es)
print(f"  type={type(es).__name__} id={hex(addr)}", flush=True)
print(f"  回收前：w() is es = {w() is es}", flush=True)
del es
gc.collect()
got = w()
print(f"  回收后（del + gc.collect）：w() = {got!r}", flush=True)
if got is None:
    print("  ⇒ ✅ 正确置死（与 Stream 一致）", flush=True)
else:
    print(f"  ⇒ ❌ **未置死**：weakref 悬垂，指向类型 {type(got).__name__} (id={hex(id(got))})", flush=True)

# 再分配若干对象，看是否落在同一地址、w() 会"跟过去"
hit = None
for i in range(300):
    t = torch.cuda.Stream()
    if id(t) == addr:
        hit = (i + 1, t)
        break
if hit:
    n, t = hit
    held = w()
    print(f"  第 {n} 个新对象落在同一地址 {hex(addr)}："
          f"w() 类型={type(held).__name__ if held is not None else None} "
          f"w() is t = {w() is t}", flush=True)
    if w() is t:
        print("  ⇒ ⇒ **根因复现**：弱引用指向了『同地址的新对象』⇒ 归属判定必然误判", flush=True)
else:
    print("  （300 次分配内未出现同地址复用）", flush=True)

# 对照：普通 Python 类实例
class Foo:
    pass


f = Foo()
wf = weakref.ref(f)
del f
gc.collect()
print(f"\n=== 对照：普通类实例 ===\n  回收后 wf() = {wf()!r}", flush=True)
