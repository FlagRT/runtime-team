#!/usr/bin/env python3
"""实验：hcclAdaptorAllGather 检查当前设备 + 用全新 aclrtCreateStream 流"""
P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"
s = open(P).read()

old = """  // EXPERIMENT: use the current default stream instead of the passed-in one
  aclrtStream useStream = stream ? stream->base : nullptr;
  {
    aclrtStream cur = nullptr;
    aclError ce = aclrtCtxGetCurrentDefaultStream(&cur);
    if (ce == 0 && cur) useStream = cur;
    fprintf(stderr, "[HCcLDBG] using stream: pass=%p default=%p\\n",
            (void *)(stream ? stream->base : 0), (void *)cur);
  }
  HcclResult agRet = HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, useStream);"""
new = """  // EXPERIMENT2: check current device and use a brand-new stream
  int curDev = -1;
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
assert s.count(old) == 1, "experiment block not found"
s = s.replace(old, new, 1)
open(P, "w").write(s)
print("OK: 实验2 已应用（检查设备 + 全新流）")
