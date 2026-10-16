"""用**真实的 base.py 代码**判定：「弱引用 + 身份复核」在 id 复用下是否会越权销毁。

做法：动态构造一个最小后端（跳过 ABC 抽象检查），
  · 以**弱引用**方式登记一个对象（等价于修复前的 `_register_owned_stream`）
  · 让该对象被回收
  · 反复分配同尺寸对象，直到拿到同一个 `id`（等价于"新流撞上旧条目的 id"）
  · 调**真实的** `release_stream()`
"""
import gc
import sys

sys.path.insert(0, "/Users/hliu553/WorkBuddy/Flag/runtime-team-gh/dev/device-context/prototype")
from runtime.backends.base import RuntimeBackend       # noqa: E402


class Fake(RuntimeBackend):
    name = "fake"
    device_type = "fake"

    def _destroy_stream_raw(self, handle) -> bool:
        print(f"    ⚠️ 真的调用了销毁原语，handle={handle:#x}（= 越权销毁）")
        return True


Fake.__abstractmethods__ = frozenset()                  # 跳过 ABC 检查
bk = Fake()

print("=== 场景一：弱引用登记 ⇒ 对象被回收 ⇒ 新对象复用其 id ===")
A = object()
idA = id(A)
bk._reg("_owned_streams")[idA] = (0xDEAD, bk._id_holder(A))
print(f"  登记: id(A)={idA:#x} holder 类型={type(bk._id_holder(A)).__name__}")
del A
gc.collect()

B = None
for _ in range(200000):
    cand = object()
    if id(cand) == idA:
        B = cand
        break
if B is None:
    print("  ❌ 未能构造 id 复用（无法判定）")
else:
    print(f"  id 复用成功: id(B)={id(B):#x} == id(A)")
    r = bk.release_stream(B)
    print(f"  release_stream(B) = {r!r}   （契约期望 False）")
    print(f"  ⇒ {'✅ 弱引用下被正确拦下（不会越权销毁）' if r is False else '❌ 越权销毁'}")

print()
print("=== 场景二：同一对象（真拥有）⇒ 必须释放并返回 True ===")
bk2 = Fake()
C = object()
bk2._reg("_owned_streams")[id(C)] = (0xBEEF, bk2._id_holder(C))
print(f"  release_stream(C) = {bk2.release_stream(C)!r}   （期望 True）")
