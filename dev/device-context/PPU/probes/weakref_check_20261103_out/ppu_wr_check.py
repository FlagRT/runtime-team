import sys, weakref
mode = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
import torch
torch.cuda.set_device(0)
refs = []
for _ in range(N):
    if mode == "wr_tensor":
        o = torch.zeros(8, device="cuda:0")
    else:
        o = torch.cuda.Stream()
    if mode in ("wr_stream", "wr_tensor"):
        refs.append(weakref.ref(o))
    if mode != "wr_hold":
        del o
print(f"[{mode}] runtime_imported=False N={N} refs={len(refs)} strong={'yes' if mode=='wr_hold' else 'no'}", flush=True)
print(f"[{mode}] reached end, normal exit ...", flush=True)
