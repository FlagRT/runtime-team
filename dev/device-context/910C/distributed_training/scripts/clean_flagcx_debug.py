#!/usr/bin/env python3
"""清理 FlagCX 中的临时 debug 打印（HCcLDBG/AGDBG）"""
import re

files = {
    "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc": [
        r'  fprintf\(stderr, "\[HCcLDBG\] CommInitRank[^\n]*\n',
        r'  fprintf\(stderr, "\[HCcLDBG\] AllGather[^\n]*\n',
        r'  fprintf\(stderr, "\[HCcLDBG\] HcclAllGather[^\n]*\n',
        r'  fprintf\(stderr, "\[HCcLDBG\] allgather cur dev[^\n]*\n',
    ],
    "/workspace/FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp": [
        r'  fprintf\(stderr, "\[AGDBG\][^\n]*\n',
    ],
}

for path, pats in files.items():
    s = open(path).read()
    for pat in pats:
        s2 = re.sub(pat, '', s)
        if s2 != s:
            print(f"[{path.split('/')[-1]}] removed: {pat[:50]}...")
        s = s2
    open(path, "w").write(s)
print("=== debug 打印清理完成 ===")
