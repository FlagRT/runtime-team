"""MLU590 步 0：验收模型文件完整性核验（非有限值扫描）。

用法：python3 mlu_model_integrity.py <模型目录>
只读；不写任何文件。
"""
import os
import sys

import torch
from safetensors import safe_open

P = sys.argv[1] if len(sys.argv) > 1 else (
    "/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
)
print("模型目录       =", P)
print("realpath       =", os.path.realpath(P))
print("is symlink     =", os.path.islink(P))
print("目录内容:")
for f in sorted(os.listdir(P)):
    fp = os.path.join(P, f)
    print("   %12d  %s" % (os.path.getsize(fp), f))

sf = os.path.join(P, "model.safetensors")
print("\n--- safetensors 逐张量非有限值扫描 ---")
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

print("\nVERDICT =", "MODEL_INTEGRITY_PASS" if not bad else "MODEL_INTEGRITY_FAIL")
