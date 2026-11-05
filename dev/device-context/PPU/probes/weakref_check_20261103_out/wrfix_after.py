import sys
import torch            # 先 import torch，确保同类先建好
import wrfix            # import 即打补丁
info = wrfix.info()
print("wrfix.info =", info, flush=True)
torch.cuda.set_device(0)

# P1：回调是否开始触发（修前 Stream/Event 均为 False）
import weakref, gc
for name, factory in (("Stream", torch.cuda.Stream), ("Event", torch.cuda.Event)):
    called = []
    o = factory()
    keep = weakref.ref(o, lambda _r: called.append(True))
    del o
    gc.collect()
    print(f"{name}_callback_fired={len(called) == 1} (called={len(called)})", flush=True)
    del keep

# P3：功能形态不变 —— 流可用 / 事件计时可用 / 张量运算正确
s = torch.cuda.Stream()
e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
with torch.cuda.stream(s):
    e1.record()
    x = torch.ones(256, 256, device="cuda:0")
    y = (x * 3).sum()
    e2.record()
torch.cuda.synchronize()
val = float(y)
ms = e1.elapsed_time(e2)
ok_val = abs(val - 3.0 * 256 * 256) < 1e-3
print(f"SANITY sum={val} expect={3.0*256*256} ok={ok_val}", flush=True)
print(f"SANITY elapsed_ms={ms:.3f} plausible={0.0 <= ms < 5000}", flush=True)
print(f"SANITY verdict={'PASS' if ok_val and 0.0 <= ms < 5000 else 'FAIL'}", flush=True)
