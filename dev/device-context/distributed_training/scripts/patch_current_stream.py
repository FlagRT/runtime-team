#!/usr/bin/env python3
"""backend_flagcx.cpp: getStreamByIndex(0) 返回调用者当前 stream（PyTorch 标准语义）"""
P = "/workspace/FlagCX/plugin/torch/flagcx/src/backend_flagcx.cpp"
s = open(P).read()

# 1. 顶部加 extern 声明（GetCurrentStream/SetCurrentStream 由 torch_fl 暴露）
old = """#include <nlohmann/json.hpp>
#include <stdexcept>"""
new = """#include <nlohmann/json.hpp>
#include <stdexcept>

// torch_fl 暴露的当前 stream 切换 API（csrc/runtime/accelerator/ascend/stream_api.cc）
// 用于让 FlagCX collective 在调用者的当前 stream 上执行（PyTorch 标准语义）。
extern "C" void *GetCurrentStream(int device_index);"""
assert s.count(old) == 1, "include block not found"
s = s.replace(old, new, 1)

# 2. getStreamByIndex：streamId==0 返回当前 stream（不缓存）
old2 = """flagcxStream_t flagcxBackend::getStreamByIndex(int streamId) {
  if (auto search = flagcxStreams_.find(streamId);
      search != flagcxStreams_.end()) {
    return search->second;
  } else {
    flagcxStreams_[streamId] = nullptr;
    // FlagOS adaptation: use flagcx device-handle stream creation (ACL-based),
    // replacing torch_npu's c10_npu stream on Ascend.
    C10D_FLAGCX_CHECK(devHandle_->streamCreate(&flagcxStreams_[streamId]),
                      std::nullopt);
    return flagcxStreams_[streamId];
  }
}"""
new2 = """flagcxStream_t flagcxBackend::getStreamByIndex(int streamId) {
  if (streamId == 0) {
    // Kistich: run collectives on the caller's CURRENT stream (PyTorch
    // semantics) so results are visible to subsequent tensor ops. Using an
    // internal cached stream breaks happens-before: HcclAllGather submits to
    // one stream while tensor reads synchronize another -> stale/zero data.
    static thread_local flagcxStream curStream;
    curStream.base = (aclrtStream)GetCurrentStream(deviceId_);
    return &curStream;
  }
  if (auto search = flagcxStreams_.find(streamId);
      search != flagcxStreams_.end()) {
    return search->second;
  } else {
    flagcxStreams_[streamId] = nullptr;
    // FlagOS adaptation: use flagcx device-handle stream creation (ACL-based),
    // replacing torch_npu's c10_npu stream on Ascend.
    C10D_FLAGCX_CHECK(devHandle_->streamCreate(&flagcxStreams_[streamId]),
                      std::nullopt);
    return flagcxStreams_[streamId];
  }
}"""
assert s.count(old2) == 1, "getStreamByIndex not found"
s = s.replace(old2, new2, 1)

open(P, "w").write(s)
print("OK: getStreamByIndex(0) 改用调用者当前 stream")
