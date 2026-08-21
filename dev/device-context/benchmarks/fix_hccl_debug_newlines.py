#!/usr/bin/env python3
"""修复 hccl_adaptor.cc 被 heredoc 破坏的 fprintf 换行 + 确保 debug 打印完整"""
import sys

P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"

fixes = [
    # (损坏原文, 正确替换) —— 损坏处是 fprintf 字符串里 \n 被炸成真实换行
    (
        '  fprintf(stderr, "[HCcLDBG] CommInitRank rank=%d nranks=%d ret=%d base=%p\n",\n',
        '  fprintf(stderr, "[HCcLDBG] CommInitRank rank=%d nranks=%d ret=%d base=%p\\n",\n',
    ),
    (
        '  fprintf(stderr, "[HCcLDBG] AllGather sendcount=%zu comm=%p base=%p stream=%p sbase=%p\n",\n',
        '  fprintf(stderr, "[HCcLDBG] AllGather sendcount=%zu comm=%p base=%p stream=%p sbase=%p\\n",\n',
    ),
    (
        '  fprintf(stderr, "[HCcLDBG] HcclAllGather ret=%d sendptr=%p recvptr=%p\n",\n',
        '  fprintf(stderr, "[HCcLDBG] HcclAllGather ret=%d sendptr=%p recvptr=%p\\n",\n',
    ),
]

s = open(P).read()
for old, new in fixes:
    # 文件里实际是 "..." + 真实换行 + '",'（即 \n 和 " 之间断了行）
    broken_variant = old.replace("\\n\",\n", '\\n" + "\n",\n')
    # 用更宽松的匹配：找 "..."换行 + '"' 的结构
    import re
    # 匹配: fprintf(stderr, "..." 后跟真实换行, 然后独立一行 ",
    # 将 换行+" 替换为 \n"
    pat = None
    for m in re.finditer(r'fprintf\(stderr, "([^"]*)"\n",\n', s):
        content = m.group(1)
        if "HCcLDBG" in content:
            fixed = 'fprintf(stderr, "%s\\n",\n' % content
            s = s[:m.start()] + fixed + s[m.end():]
            print(f"[FIX] fprintf: {content[:50]}...")
            break

open(P, "w").write(s)
print("=== 完成，验证 ===")
for i, line in enumerate(open(P).readlines(), 1):
    if "HCcLDBG" in line:
        print(i, line.rstrip()[:90])
