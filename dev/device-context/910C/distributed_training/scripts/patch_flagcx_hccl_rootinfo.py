#!/usr/bin/env python3
"""
FlagCX 双卡修复：HcclRootInfo(4108B) > flagcxUniqueId(256B) 缓冲区溢出与截断
================================================================================
根因：
  - hcclAdaptorGetUniqueId 把 4108B 的 HcclRootInfo 直接写入 256B 的 flagcxUniqueId
    → 缓冲区溢出 + allGather 只交换 256B → 其他 rank 拿到残缺 RootInfo
    → HcclCommInitRootInfo 解析失败 → flagcxInvalidArgument(4)

修复：
  1) flagcx.cc  flagcxHomoCommInit：commInitRank 调用传入 bootstrap state（原为 NULL）
  2) hccl_adaptor.cc：
     - GetUniqueId：RootInfo 存 thread_local，uniqueId 仅作已就绪标记（不再溢出）
     - CommInitRank：rank0 生成 RootInfo → bootstrapCollBroadcast 全量分发 4108B
       → 全员用完整 RootInfo 调 HcclCommInitRootInfo
"""
import sys

FLAGCX_CC = "/workspace/FlagCX/flagcx/flagcx.cc"
HCCL_ADAPTOR_CC = "/workspace/FlagCX/flagcx/adaptor/ccl/hccl_adaptor.cc"


def patch_flagcx_cc():
    s = open(FLAGCX_CC).read()
    old = """  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commInitRank(
      homoComm, comm->homoRanks, commId, comm->homoRank, NULL));"""
    new = """  // Pass the bootstrap state so the ccl adaptor can exchange
  // vendor-specific root info (e.g. 4108B HcclRootInfo) that does not
  // fit inside the 256B flagcxUniqueId.
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commInitRank(
      homoComm, comm->homoRanks, commId, comm->homoRank, state));"""
    assert s.count(old) == 1, f"flagcx.cc: expected 1 match, got {s.count(old)}"
    s = s.replace(old, new)
    open(FLAGCX_CC, "w").write(s)
    print(f"[OK] {FLAGCX_CC}: commInitRank 传入 bootstrap state")


def patch_hccl_adaptor():
    s = open(HCCL_ADAPTOR_CC).read()

    # ---- 1. 在文件头（USE_ASCEND_ADAPTOR 内）加 thread_local 存储 ----
    anchor = """#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include <map>
#include <vector>"""
    assert anchor in s, "hccl_adaptor.cc: include 段未找到"
    storage = anchor + """

// HcclRootInfo is 4108B while flagcxUniqueId is only 256B. Keep the root info
// in thread-local storage and use the uniqueId buffer merely as a marker.
// (Kistich: fix for dual-GPU HCCL comm init InvalidArgument)
static thread_local HcclRootInfo t_hcclRootInfo;
static thread_local bool t_hasHcclRootInfo = false;"""
    s = s.replace(anchor, storage, 1)

    # ---- 2. 重写 hcclAdaptorGetUniqueId（防溢出）----
    old_get = """flagcxResult_t hcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return (
      flagcxResult_t)h2f_ret_map[HcclGetRootInfo((HcclRootInfo *)(*uniqueId))];
}"""
    new_get = """flagcxResult_t hcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  HcclResult ret = HcclGetRootInfo(&t_hcclRootInfo);
  if (ret != HCCL_SUCCESS) {
    return (flagcxResult_t)h2f_ret_map[ret];
  }
  t_hasHcclRootInfo = true;
  // flagcxUniqueId(256B) cannot hold HcclRootInfo(4108B); store a marker only.
  memset((void *)(*uniqueId), 0, sizeof(**uniqueId));
  memcpy((void *)(*uniqueId), "HCCLROOT", 8);
  return flagcxSuccess;
}"""
    assert s.count(old_get) == 1, "GetUniqueId 原文未找到或重复"
    s = s.replace(old_get, new_get, 1)

    # ---- 3. 重写 hcclAdaptorCommInitRank（bootstrap 全量分发 RootInfo）----
    old_init = """flagcxResult_t hcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       struct bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)h2f_ret_map[HcclCommInitRootInfo(
      nranks, (HcclRootInfo *)commId, rank, &(*comm)->base)];
}"""
    new_init = """flagcxResult_t hcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       struct bootstrapState *bootstrap) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  HcclRootInfo rootInfo;
  memset(&rootInfo, 0, sizeof(rootInfo));
  if (bootstrap != NULL) {
    // Rank 0 generates the root info (thread-local from GetUniqueId, or fresh)
    // and broadcasts the full 4108B to every rank over the bootstrap network.
    if (rank == 0) {
      if (!t_hasHcclRootInfo) {
        HcclResult ret = HcclGetRootInfo(&t_hcclRootInfo);
        if (ret != HCCL_SUCCESS) {
          return (flagcxResult_t)h2f_ret_map[ret];
        }
        t_hasHcclRootInfo = true;
      }
      memcpy(&rootInfo, &t_hcclRootInfo, sizeof(HcclRootInfo));
    }
    FLAGCXCHECK(bootstrapCollBroadcast(bootstrap, rank, nranks, 0, &rootInfo,
                                       sizeof(HcclRootInfo)));
  } else if (t_hasHcclRootInfo) {
    // No bootstrap available (e.g. single process): reuse thread-local root
    // info produced by hcclAdaptorGetUniqueId in the same process.
    memcpy(&rootInfo, &t_hcclRootInfo, sizeof(HcclRootInfo));
  } else {
    HcclResult ret = HcclGetRootInfo(&rootInfo);
    if (ret != HCCL_SUCCESS) {
      return (flagcxResult_t)h2f_ret_map[ret];
    }
  }
  return (flagcxResult_t)h2f_ret_map[HcclCommInitRootInfo(
      nranks, &rootInfo, rank, &(*comm)->base)];
}"""
    assert s.count(old_init) == 1, "CommInitRank 原文未找到或重复"
    s = s.replace(old_init, new_init, 1)

    open(HCCL_ADAPTOR_CC, "w").write(s)
    print(f"[OK] {HCCL_ADAPTOR_CC}: GetUniqueId 防溢出 + CommInitRank bootstrap 分发")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--verify-only":
        s = open(HCCL_ADAPTOR_CC).read()
        print("t_hasHcclRootInfo 出现次数:", s.count("t_hasHcclRootInfo"))
        print("bootstrapCollBroadcast 出现次数:", s.count("bootstrapCollBroadcast"))
        sys.exit(0)
    patch_flagcx_cc()
    patch_hccl_adaptor()
    print("\n=== 全部 patch 完成 ===")
