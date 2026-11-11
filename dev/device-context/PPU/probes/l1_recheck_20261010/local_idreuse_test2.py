"""用**真实的 base.py 代码**判定：「弱引用 + 身份复核」在 id 复用下是否会越权销毁。

上一版用 `object()` 做样本 —— 它**不可弱引用**（无 `__weakref__` 槽）⇒ `_id_holder` 走了
TypeError 回退（强持有）⇒ 样本选错。本版改用**普通类实例**（可弱引用），
并同时报告两种样本下 `_id_holder` 的真实分支。
"""
import gc
import inspect
import sys
import weakref

sys.path.insert(0, "/Users/hliu553/WorkBuddy/Flag/runtime-team-gh/dev/device-context/prototype")
from runtime.backends.base import RuntimeBackend       # noqa: E402

print("=== _id_holder 的实际源码（运行中的那份）===")
print(inspect.getsource(RuntimeBackend._id_holder))


class Foo:
    pass


print("=== 弱引用可用性 ===")
for maker, label in ((object, "object()"), (Foo, "class Foo 实例")):
    x = maker()
    try:
        r = weakref.ref(x)
        print(f"  {label:16s} weakref.ref 可用 → {type(r).__name__}")
    except TypeError as e:
        print(f"  {label:16s} weakref.ref 抛 TypeError: {e}")


class Fake(RuntimeBackend):
    name = "fake"
    device_type = "fake"

    def _destroy_stream_raw(self, handle) -> bool:
        print(f"    ⚠️ **真的调用了销毁原语**，handle={handle:#x}（= 越权销毁）")
        return True


Fake.__abstractmethods__ = frozenset()
bk = Fake()

print()
print("=== 场景一：弱引用登记 ⇒ 对象被回收 ⇒ 新对象复用其 id ⇒ release_stream 应返回 False ===")
A = Foo()
idA = id(A)
holder = bk._id_holder(A)
print(f"  登记: id(A)={idA:#x} holder 类型={type(holder).__name__}")
bk._reg("_owned_streams")[idA] = (0xDEAD, holder)
del A
gc.collect()
print(f"  回收后 holder() = {holder()!r}（弱引用应为 None）")

B = None
for _ in range(300000):
    cand = Foo()
    if id(cand) == idA:
        B = cand
        break
if B is None:
    print("  ❌ 未能构造 id 复用（无法判定）")
else:
    print(f"  id 复用成功: id(B)={id(B):#x} == id(A)")
    r = bk.release_stream(B)
    print(f"  release_stream(B) = {r!r}   （契约期望 False）")
    verdict = ("✅ 弱引用下身份复核**正确拦下**（不会越权销毁）"
               if r is False else "❌ 越权销毁 —— 身份复核失效")
    print(f"  ⇒ {verdict}")

print()
print("=== 场景二：同一对象（真拥有）⇒ 必须释放并返回 True ===")
bk2 = Fake()
C = Foo()
bk2._reg("_owned_streams")[id(C)] = (0xBEEF, bk2._id_holder(C))
print(f"  release_stream(C) = {bk2.release_stream(C)!r}   （期望 True）")
