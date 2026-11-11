"""在 910C 上把补丁包应用到「坏文件的副本」，产出一个待验证的修复文件。

用法: python3 apply_patch.py <坏文件> <输出文件> <补丁包>
"""

import shutil
import struct
import sys

src, dst, bundle = sys.argv[1], sys.argv[2], sys.argv[3]
shutil.copy2(src, dst)

with open(bundle, "rb") as f:
    cnt = struct.unpack("<Q", f.read(8))[0]
    with open(dst, "r+b") as g:
        for i in range(cnt):
            off, ln = struct.unpack("<QI", f.read(12))
            g.seek(off)
            g.write(f.read(ln))
print("applied %d segments" % cnt)
