import torch
import wrfix      # 打补丁
torch.cuda.set_device(0)
import weakref
refs = []
for _ in range(30):
    s = torch.cuda.Stream()
    refs.append(weakref.ref(s))
    del s
print("wrfix applied, refs=%d" % len(refs), flush=True)
print("reached end, normal exit ...", flush=True)
