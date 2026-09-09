#!/usr/bin/env python3
"""修复 hccl_adaptor.cc 清理 debug 后残留的碎行"""
P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"
s = open(P).read()

removals = [
    # CommInitRank 残渣（fprintf 参数行）
    "          rank, nranks, (int)initRet, (void *)(*comm)->base);\n",
    # AllGather 残渣（fprintf 参数行 1+2）
    "          sendcount, (void *)comm, (void *)(comm ? comm->base : 0),\n",
    "          (void *)stream, (void *)(stream ? stream->base : 0));\n",
    # AllGather 调试变量段
    "  // EXPERIMENT2: check current device and use a brand-new stream\n",
    "  int curDev = -1;\n",
    "  aclrtGetDevice(&curDev);\n",
    "          (void *)(stream ? stream->base : nullptr));\n",
    # AllGather 尾部 fprintf 参数行
    "          (int)agRet, sendbuffptr, recvbuff);\n",
]

for r in removals:
    cnt = s.count(r)
    if cnt != 1:
        print(f"WARN: removal not found or dup ({cnt}): {r[:60]!r}")
    s = s.replace(r, "", 1)

open(P, "w").write(s)
print("残渣清理完成")

# 验证残留模式
import re
bad = re.findall(r"^\s+[a-z].*\);$", s, re.M)
print("可疑残留行数:", len(bad))
