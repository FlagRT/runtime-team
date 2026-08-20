#!/usr/bin/env python3
"""Patch stream_guard_flagcx.hpp remaining c10_npu usages -> aclrtStream."""
import sys

P = "/workspace/FlagCX/plugin/torch/flagcx/include/stream_guard_flagcx.hpp"
s = open(P).read()
FAILED = []

def rep(old, new, label):
    global s
    if old not in s:
        print(f"[{label}] NOT FOUND")
        FAILED.append(label)
        return
    s = s.replace(old, new)
    print(f"[{label}] OK")

# constructor guard
rep('''#elif USE_ASCEND_ADAPTOR
        guard_(c10_npu::getNPUStreamFromPool(deviceId))''',
    '''#elif USE_ASCEND_ADAPTOR
        guard_(*(aclrtStream *)stream)''',
    "ctor")

# reset_stream
rep('''#elif USE_ASCEND_ADAPTOR
    guard_ = c10_npu::getNPUStreamFromPool(deviceId_);''',
    '''#elif USE_ASCEND_ADAPTOR
    guard_ = *(aclrtStream *)stream;''',
    "reset_stream")

# member
rep('''#elif USE_ASCEND_ADAPTOR
  c10_npu::NPUStream guard_;''',
    '''#elif USE_ASCEND_ADAPTOR
  aclrtStream guard_;''',
    "member")

if FAILED:
    print(f"INCOMPLETE: {FAILED}")
    sys.exit(1)
open(P, "w").write(s)
print("ALL STREAM_GUARD PATCHES APPLIED")
