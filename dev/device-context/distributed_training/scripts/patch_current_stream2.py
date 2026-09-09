#!/usr/bin/env python3
"""backend_flagcx.cpp: 修正 getStreamByIndex(0) 实现（避免 flagcxStream 完整类型）"""
P = "/workspace/FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp"
s = open(P).read()

old = """  if (streamId == 0) {
    // Kistich: run collectives on the caller's CURRENT stream (PyTorch
    // semantics) so results are visible to subsequent tensor ops. Using an
    // internal cached stream breaks happens-before: HcclAllGather submits to
    // one stream while tensor reads synchronize another -> stale/zero data.
    static thread_local flagcxStream curStream;
    curStream.base = (aclrtStream)GetCurrentStream(deviceId_);
    return &curStream;
  }"""
new = """  if (streamId == 0) {
    // Kistich: run collectives on the caller's CURRENT stream (PyTorch
    // semantics) so results are visible to subsequent tensor ops. Using an
    // internal cached stream breaks happens-before: HcclAllGather submits to
    // one stream while tensor reads synchronize another -> stale/zero data.
    // flagcxStream { aclrtStream base; } -> first member is the ACL stream,
    // so a pointer to our void* storage doubles as a flagcxStream*.
    static thread_local void *curAclStream = nullptr;
    curAclStream = GetCurrentStream(deviceId_);
    return (flagcxStream_t)&curAclStream;
  }"""
assert s.count(old) == 1, "getStreamByIndex block not found"
s = s.replace(old, new, 1)
open(P, "w").write(s)
print("OK: getStreamByIndex(0) 已修正（void* 布局技巧）")
