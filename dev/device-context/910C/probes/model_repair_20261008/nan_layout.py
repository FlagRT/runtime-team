"""判定 NaN 的**空间分布**：连续块（⇒ 文件/块设备损坏）还是散布（⇒ 权重本身）。"""

import json
import struct

P = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B/model.safetensors"

with open(P, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

for name in ("layers.1.self_attn.v_proj.weight", "embed_tokens.weight"):
    meta = hdr[name]
    base = 8 + n + meta["data_offsets"][0]
    nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(P, "rb") as f:
        f.seek(base)
        raw = f.read(nbytes)

    nan_idx = []
    for i in range(0, len(raw) - 1, 2):
        v = raw[i] | (raw[i + 1] << 8)
        if (v & 0x7FFF) > 0x7F80:
            nan_idx.append(i // 2)

    print("=== %s ===" % name)
    print("  NaN 元素数 =", len(nan_idx), "/", len(raw) // 2)
    if not nan_idx:
        continue
    # 连续段
    runs = []
    s = p = nan_idx[0]
    for x in nan_idx[1:]:
        if x == p + 1:
            p = x
        else:
            runs.append((s, p))
            s = p = x
    runs.append((s, p))
    print("  连续段数 =", len(runs), " 最长段 =", max(b - a + 1 for a, b in runs))
    print("  前 8 段（元素索引区间）:", runs[:8])
    print("  后 4 段:", runs[-4:])
    # 该段对应的文件字节偏移（相对文件绝对偏移）
    a, b = runs[0]
    abs_off = base + a * 2
    with open(P, "rb") as f:
        f.seek(max(0, abs_off - 32))
        ctx = f.read(32 + (b - a + 1) * 2 + 32)
    print("  首段邻域字节（前 32B + 段内 + 后 32B）:")
    print("   ", ctx[:32].hex(" "))
    print("   ", ctx[32:32 + (b - a + 1) * 2].hex(" "))
    print("   ", ctx[32 + (b - a + 1) * 2:].hex(" "))
    print("  段起始文件绝对偏移 =", abs_off, " (mod 4096 =", abs_off % 4096, ")")
