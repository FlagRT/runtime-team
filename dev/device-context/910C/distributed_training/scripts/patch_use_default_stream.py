#!/usr/bin/env python3
"""实验：hcclAdaptorAllGather 改用当前默认流（验证 torch_fl stream 是否与 HCCL 不兼容）"""
P = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"
s = open(P).read()

old = """  fprintf(stderr, "[HCcLDBG] AllGather sendcount=%zu comm=%p base=%p stream=%p sbase=%p\\n",
          sendcount, (void *)comm, (void *)(comm ? comm->base : 0),
          (void *)stream, (void *)(stream ? stream->base : 0));
  HcclResult agRet = HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, stream->base);"""
new = """  fprintf(stderr, "[HCcLDBG] AllGather sendcount=%zu comm=%p base=%p stream=%p sbase=%p\\n",
          sendcount, (void *)comm, (void *)(comm ? comm->base : 0),
          (void *)stream, (void *)(stream ? stream->base : 0));
  // EXPERIMENT: use the current default stream instead of the passed-in one
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
assert s.count(old) == 1, "allgather body not found"
s = s.replace(old, new, 1)
open(P, "w").write(s)
print("OK: allgather 改用默认流")
