"""在 910C 上算「损坏页清单」：把含 NaN 的 4KB 页全部列出（合并相邻页）。

输出 JSON: {"pages": [[off, len], ...], "total_bytes": N}
"""

import json
import struct
import sys

import numpy as np

P = sys.argv[1]
OUT = sys.argv[2]
PG = 4096

with open(P, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base = 8 + n

pages = set()
detail = {}
for name, meta in hdr.items():
    if name == "__metadata__" or meta["dtype"] != "BF16":
        continue
    off0, off1 = meta["data_offsets"]
    with open(P, "rb") as f:
        f.seek(base + off0)
        raw = f.read(off1 - off0)
    v = np.frombuffer(raw, dtype="<u2")
    bad = np.nonzero((v & 0x7FFF) > 0x7F80)[0]
    if bad.size == 0:
        continue
    detail[name] = int(bad.size)
    # NaN 元素索引 → 文件绝对字节偏移 → 所属 4KB 页
    for idx in bad:
        a = base + off0 + int(idx) * 2
        pages.add(a // PG * PG)

# 合并相邻页
spans = []
ps = sorted(pages)
s = e = ps[0]
for p in ps[1:]:
    if p == e + PG:
        e = p
    else:
        spans.append([s, e + PG - s])
        s = e = p
spans.append([s, e + PG - s])

# 文件头区（safetensors header + 前置长度）必须一并核对
spans = [[0, base]] + spans
spans.sort()

total = sum(x[1] for x in spans)
print("含 NaN 的张量:", detail)
print("页数:", len(pages), " 合并后段数:", len(spans), " 需取字节:", total, "(%.2f MB)" % (total / 1e6))
json.dump({"pages": spans, "total_bytes": total}, open(OUT, "w"))
print("→", OUT)
