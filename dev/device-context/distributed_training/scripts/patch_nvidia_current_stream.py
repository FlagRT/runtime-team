#!/usr/bin/env python3
"""4090 nvidia 分支 stream 语义修复：collective 跑在 torch 当前 CUDA 流。

背景：getStreamByIndex(0) 在 nvidia 分支走自建缓存流，与 tensor 计算
所在的 torch 当前流无 happens-before -> 双卡 allgather 第二次调用
rank1 数据全 0（910C 昇腾分支同根因的 nvidia 变体）。
修复：streamId==0 时返回 torch 当前 CUDA 流（at::cuda::getCurrentCUDAStream），
flagcxStream { cudaStream_t base; } 结构 => 指向 thread_local cudaStream_t
变量的指针可直接当 flagcxStream_t 用（base = 当前流）。
"""
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "/home/data/hongbinliu/FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp"
s = open(p).read()

# 1) include
if "#include <ATen/cuda/CUDAContext.h>" not in s:
    anchor = '#include "backend_flagcx.hpp"'
    assert anchor in s, "include anchor not found"
    s = s.replace(anchor, anchor + "\n#include <ATen/cuda/CUDAContext.h>", 1)

# 2) getStreamByIndex 开头加 nvidia 当前流分支
old = """flagcxStream_t flagcxBackend::getStreamByIndex(int streamId) {
#ifdef USE_ASCEND_ADAPTOR"""
new = """flagcxStream_t flagcxBackend::getStreamByIndex(int streamId) {
#ifdef USE_NVIDIA_ADAPTOR
  if (streamId == 0) {
    // 4090 line: run collectives on the CURRENT CUDA stream (PyTorch
    // semantics) so results are visible to subsequent tensor ops on the
    // same stream. A self-managed cached stream breaks happens-before:
    // NCCL submits on one stream while tensor reads synchronize another
    // -> stale/zero data (same root cause fixed on the ascend line).
    static thread_local cudaStream_t curCudaStream = nullptr;
    curCudaStream = at::cuda::getCurrentCUDAStream(deviceId_).stream();
    return reinterpret_cast<flagcxStream_t>(&curCudaStream);
  }
#endif
#ifdef USE_ASCEND_ADAPTOR"""
assert old in s, "getStreamByIndex anchor not found"
s = s.replace(old, new, 1)
open(p, "w").write(s)
print("patched OK:", p)
