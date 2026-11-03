import sys, weakref
NS = sys.argv[1] if len(sys.argv) > 1 else "cuda"
import torch
if NS == "cuda":
    torch.cuda.set_device(0); S = torch.cuda.Stream
else:
    import torch_mlu; torch.mlu.set_device(0); S = torch.mlu.Stream
refs = []
for _ in range(30):
    s = S(); refs.append(weakref.ref(s)); del s
print("refs=%d reached end, normal exit ..." % len(refs), flush=True)
