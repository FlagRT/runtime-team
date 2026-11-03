import sys, weakref
NS = sys.argv[1] if len(sys.argv) > 1 else "cuda"
import torch
if NS == "cuda":
    torch.cuda.set_device(0); S = torch.cuda.Stream
else:
    import torch_mlu  # noqa: F401
    torch.mlu.set_device(0); S = torch.mlu.Stream
refs = []
made = 0
for _ in range(30):
    s = S(); made += 1
    try:
        refs.append(weakref.ref(s))
    except TypeError:
        pass
    del s
print("made=%d weakrefable=%d" % (made, len(refs)), flush=True)
print("reached end, normal exit ...", flush=True)
