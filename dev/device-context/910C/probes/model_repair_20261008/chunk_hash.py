"""分块 sha256 比对：只交换哈希，不传数据。

用法: python3 chunk_hash.py <file> <chunk_bytes> [start] [end]
输出: 每行 "<chunk_index> <sha256>"
"""

import hashlib
import sys

path = sys.argv[1]
chunk = int(sys.argv[2])
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
end = int(sys.argv[4]) if len(sys.argv) > 4 else None

with open(path, "rb") as f:
    f.seek(start)
    idx = start // chunk
    pos = start
    while end is None or pos < end:
        n = min(chunk, (end - pos) if end is not None else chunk)
        if n <= 0:
            break
        data = f.read(n)
        if not data:
            break
        print("%d %s" % (idx, hashlib.sha256(data).hexdigest()))
        idx += 1
        pos += len(data)
