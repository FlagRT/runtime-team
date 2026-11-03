import torch, weakref, gc
torch.cuda.set_device(0)
for name, factory in (("Stream", torch.cuda.Stream), ("Event", torch.cuda.Event)):
    called = []
    o = factory()
    keep = weakref.ref(o, lambda _r: called.append(True))
    del o; gc.collect()
    print(f"{name}_callback_fired={len(called) == 1} (called={len(called)})", flush=True)
    del keep
print("done", flush=True)
