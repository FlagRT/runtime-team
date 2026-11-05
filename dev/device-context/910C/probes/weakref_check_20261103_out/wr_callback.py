# 精确判据（对齐 Ascend 回归测试 _assert_weakref_callback_fires）：
# weakref 回调是否真被触发 —— 这是 PR #41596 修的行为；`ref()() is None` 不可靠
import torch, torch_npu, weakref, gc
torch.npu.set_device(0)
called = []
s = torch.npu.Stream()
keepalive = weakref.ref(s, lambda _r: called.append(True))
del s
gc.collect()
print("callback_fired:", len(called) == 1, f"(called={len(called)})")
r2 = torch.npu.Stream()
w2 = weakref.ref(r2)
del r2
gc.collect()
print("deref_after_free:", w2())  # None=已清；非 None=悬垂
