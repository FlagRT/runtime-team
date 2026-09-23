#!/usr/bin/env python3
"""hccl_adaptor.cc: 移除 ensureDevice 的 setDevice（避免 ctx 切换），改回用传入 stream"""
P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"
s = open(P).read()

# 1. 移除 hcclEnsureDevice 函数体（不再 setDevice）
old = """// HCCL's HcclCommInitRootInfo can reset the current ACL device to 0; the
// plugin layer asserts the device before each collective, but as defense-in-
// depth the adaptor records the device at comm init and re-asserts it here.
static flagcxResult_t hcclEnsureDevice() {
  if (t_deviceIndex >= 0) {
    aclrtSetDevice(t_deviceIndex);
  }
  return flagcxSuccess;
}"""
new = """// Device handling note: the training script binds each process to its own
// NPU via torch.flagos.set_device(local_rank) BEFORE HCCL init, so the current
// ACL context is correct and stable. We must NOT call aclrtSetDevice here:
// switching the current context after HCCL init invalidates HCCL's internal
// streams (rtStreamWaitEvent: stream not in current context, 107003)."""
assert s.count(old) == 1, "ensureDevice def not found"
s = s.replace(old, new, 1)

# 2. AllGather 改回用传入 stream（不 setDevice）
old2 = """  hcclEnsureDevice();
  int curDev = -1;
  aclrtGetDevice(&curDev);
  fprintf(stderr, "[HCcLDBG] after ensure device=%d\\n", curDev);
  HcclResult agRet = HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base,
      stream ? stream->base : nullptr);"""
new2 = """  int curDev = -1;
  aclrtGetDevice(&curDev);
  fprintf(stderr, "[HCcLDBG] allgather cur dev=%d stream=%p\\n", curDev,
          (void *)(stream ? stream->base : nullptr));
  HcclResult agRet = HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base,
      stream ? stream->base : nullptr);"""
assert s.count(old2) == 1, "allgather ensure call not found"
s = s.replace(old2, new2, 1)

open(P, "w").write(s)
print("OK: ensureDevice 已移除 setDevice，allgather 用传入 stream")
