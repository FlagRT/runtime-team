import sys, weakref, gc
NS = sys.argv[1] if len(sys.argv) > 1 else "cuda"
import torch
if NS == "cuda":
    import torch; torch.cuda.set_device(0); S, E = torch.cuda.Stream, torch.cuda.Event
elif NS == "mlu":
    import torch_mlu; torch.mlu.set_device(0); S, E = torch.mlu.Stream, torch.mlu.Event
print("torch=%s ns=%s" % (torch.__version__, NS), flush=True)
for name, factory in (("Stream", S), ("Event", E)):
    try:
        o = factory()
    except Exception as exc:
        print("%s_ctor_FAILED %r" % (name, exc), flush=True); continue
    called = []
    try:
        keep = weakref.ref(o, lambda _r: called.append(True))
    except TypeError as exc:
        print("%s_weakref_UNSUPPORTED %r" % (name, exc), flush=True); del o; continue
    del o; gc.collect()
    print("%s_callback_fired=%s (called=%d)" % (name, len(called) == 1, len(called)), flush=True)
    del keep
