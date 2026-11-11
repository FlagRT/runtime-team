#!/usr/bin/env python3
"""hccl_adaptor.cc: 记录设备 index + 所有 collective 前重设设备"""
P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"
s = open(P).read()

# 1. thread_local 加设备记录
old = """static thread_local HcclRootInfo t_hcclRootInfo;
static thread_local bool t_hasHcclRootInfo = false;"""
new = """static thread_local HcclRootInfo t_hcclRootInfo;
static thread_local bool t_hasHcclRootInfo = false;
static thread_local int t_deviceIndex = -1;

// HCCL's HcclCommInitRootInfo can reset the current ACL device to 0; the
// plugin layer asserts the device before each collective, but as defense-in-
// depth the adaptor records the device at comm init and re-asserts it here.
static flagcxResult_t hcclEnsureDevice() {
  if (t_deviceIndex >= 0) {
    aclrtSetDevice(t_deviceIndex);
  }
  return flagcxSuccess;
}"""
assert s.count(old) == 1, "thread_local block not found"
s = s.replace(old, new, 1)

# 2. CommInitRank 记录设备（在函数开头）
old2 = """flagcxResult_t hcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       struct bootstrapState *bootstrap) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }"""
new2 = """flagcxResult_t hcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       struct bootstrapState *bootstrap) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  aclrtGetDevice(&t_deviceIndex);"""
assert s.count(old2) == 1, "CommInitRank head not found"
s = s.replace(old2, new2, 1)

# 3. AllGather 开头重设设备（替换实验2的 current device 打印，保留诊断）
old3 = """  int curDev = -1;
  aclrtGetDevice(&curDev);
  fprintf(stderr, "[HCcLDBG] current device=%d\\n", curDev);
  aclrtStream newStream = nullptr;
  aclError csRet = aclrtCreateStream(&newStream);
  fprintf(stderr, "[HCcLDBG] aclrtCreateStream ret=%d new=%p\\n", (int)csRet,
          (void *)newStream);
  HcclResult agRet = HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, newStream);
  if (newStream) aclrtDestroyStream(newStream);"""
new3 = """  hcclEnsureDevice();
  int curDev = -1;
  aclrtGetDevice(&curDev);
  fprintf(stderr, "[HCcLDBG] after ensure device=%d\\n", curDev);
  HcclResult agRet = HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base,
      stream ? stream->base : nullptr);"""
assert s.count(old3) == 1, "exp2 block not found"
s = s.replace(old3, new3, 1)

open(P, "w").write(s)
print("OK: 设备记录 + collective 前重设已加")
