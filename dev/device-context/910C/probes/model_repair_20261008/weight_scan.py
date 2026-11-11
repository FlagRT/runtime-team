import os

import torch
from safetensors import safe_open

P = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B"
print("realpath      =", os.path.realpath(P))
print("is symlink    =", os.path.islink(P))
print("parent listing:")
for f in sorted(os.listdir(os.path.dirname(P))):
    print("   ", f)

sf = os.path.join(P, "model.safetensors")
print("\n--- safetensors 元数据 ---")
with safe_open(sf, framework="pt") as f:
    keys = list(f.keys())
    print("tensor 数 =", len(keys))
    bad = []
    total = 0
    for k in keys:
        t = f.get_tensor(k)
        total += t.numel()
        if not torch.isfinite(t).all():
            n_nan = int(torch.isnan(t).sum())
            n_inf = int(torch.isinf(t).sum())
            bad.append((k, tuple(t.shape), n_nan, n_inf, t.numel()))
    print("总参数量 =", total)
    print("含非有限值的张量数 =", len(bad))
    for b in bad[:15]:
        print("   ", b)
