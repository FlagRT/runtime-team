import weakref, gc
class Foo: pass
called = []
o = Foo(); keep = weakref.ref(o, lambda _r: called.append(True))
del o; gc.collect()
print(f"plain_object_callback_fired={len(called) == 1}")
