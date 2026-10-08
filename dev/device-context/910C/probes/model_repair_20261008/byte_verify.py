"""字节级核验：model.safetensors 里到底有没有 NaN（排除 safetensors/框架读错的可能）。

做法：绕开所有框架，直接用 json 解析文件头拿到张量偏移，
再对 NaN 位置做**裸文件字节读取**，打印原始字节。
bf16 的 NaN 位型 = 0x7FC0，Inf = 0x7F80。
"""

import json
import mmap
import struct

P = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B/model.safetensors"

with open(P, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

print("header 长度 =", n)
print("张量数 =", len([k for k in hdr if k != "__metadata__"]))

for name in ("layers.1.self_attn.v_proj.weight", "embed_tokens.weight"):
    meta = hdr[name]
    print("\n=== %s ===" % name)
    print("  dtype =", meta["dtype"], " shape =", meta["shape"], " offsets =", meta["data_offsets"])
    base = 8 + n + meta["data_offsets"][0]
    nbytes = meta["data_offsets"][1] - meta["data_offsets"][0]
    with open(P, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        raw = mm[base:base + nbytes]
        mm.close()
    # bf16 逐 2 字节扫，统计位型
    n_nan = n_inf = 0
    first_nan = first_inf = None
    for i in range(0, len(raw) - 1, 2):
        v = raw[i] | (raw[i + 1] << 8)
        if (v & 0x7FFF) > 0x7F80:
            n_nan += 1
            if first_nan is None:
                first_nan = i
        elif (v & 0x7FFF) == 0x7F80:
            n_inf += 1
            if first_inf is None:
                first_inf = i
    print("  逐 2 字节位型统计：NaN =", n_nan, " Inf =", n_inf, " 元素数 =", len(raw) // 2)
    if first_nan is not None:
        off = base + first_nan
        with open(P, "rb") as f:
            f.seek(off - 4)
            ctx = f.read(8)
        print("  首个 NaN 的**裸文件字节**（含 2 字节上下文） =", ctx.hex(" "))
        print("  → 该位置字节 = %02x %02x（bf16 NaN 位型应为 7f c0，Inf 为 7f 80）"
              % (ctx[4], ctx[5]))
        print("  文件内绝对偏移 =", off)
