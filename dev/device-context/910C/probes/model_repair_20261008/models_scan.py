"""快速扫描机上所有模型的 safetensors，统计非有限值数量（向量化，避免 Python 逐字节）。"""

import glob
import json
import os
import struct

import numpy as np


def scan(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        total = bad_tensors = bad_elems = 0
        worst = []
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            if dtype not in ("BF16", "F16", "F32"):
                continue
            off0, off1 = meta["data_offsets"]
            f.seek(8 + n + off0)
            raw = f.read(off1 - off0)
            if dtype == "BF16":
                v = np.frombuffer(raw, dtype="<u2")
                finite = ((v & 0x7FFF) <= 0x7F80)
            elif dtype == "F16":
                v = np.frombuffer(raw, dtype="<f2").astype(np.float32)
                finite = np.isfinite(v)
            else:
                v = np.frombuffer(raw, dtype="<f4")
                finite = np.isfinite(v)
            nbad = int((~finite).sum())
            total += v.size
            if nbad:
                bad_tensors += 1
                bad_elems += nbad
                worst.append((nbad, name, dtype, v.size))
        return total, bad_tensors, bad_elems, sorted(worst, reverse=True)[:6]


for d in sorted(glob.glob("/mnt/raid/hliu553/models/*")):
    if not os.path.isdir(d):
        continue
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        # 可能是 HF 布局
        files = sorted(glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True))
    print("=" * 70)
    print(os.path.basename(d))
    for p in files:
        sz = os.path.getsize(p) / 1e9
        total, bt, be, worst = scan(p)
        flag = "OK  " if be == 0 else "BAD "
        print("  %s %-46s %5.2f GB  元素 %d  坏张量 %d  坏元素 %d"
              % (flag, os.path.basename(p), sz, total, bt, be))
        for nbad, name, dt, sz2 in worst:
            print("        - %-42s %s  %d/%d (%.3f%%)" % (name, dt, nbad, sz2, 100.0 * nbad / sz2))
