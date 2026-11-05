# 空白对照：普通 Python 对象的 weakref 回调（应触发）——排除脚本自身问题
import weakref, gc
class Foo: pass
called = []
o = Foo()
k = weakref.ref(o, lambda _r: called.append(True))
del o
gc.collect()
print("plain_object_callback_fired:", len(called) == 1, f"(called={len(called)})")
