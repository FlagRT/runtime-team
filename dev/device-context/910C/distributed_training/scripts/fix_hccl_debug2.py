#!/usr/bin/env python3
"""修复 hccl_adaptor.cc 中 fprintf 字符串被真实换行破坏的问题"""
import re

P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"
s = open(P).read()

# 损坏格式示例（真实换行在字符串中间）:
#   fprintf(stderr, "[HCcLDBG] ... base=%p
# ",
# 修复为:
#   fprintf(stderr, "[HCcLDBG] ... base=%p\n",
count = 0
while True:
    m = re.search(r'(fprintf\(stderr, "[^"]*?)(\n",\n)', s)
    if not m:
        break
    s = s[: m.start()] + m.group(1) + '\\n",\n' + s[m.end() :]
    count += 1

open(P, "w").write(s)
print("修复 fprintf 处数:", count)

# 验证
for i, line in enumerate(s.splitlines(), 1):
    if "HCcLDBG" in line:
        print(i, line)
