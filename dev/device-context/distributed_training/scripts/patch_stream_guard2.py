#!/usr/bin/env python3
"""stream_guard_flagcx.hpp: ascend 分支移除 SetCurrentStream 调用（getStreamByIndex(0) 已是当前 stream）"""
P = "/workspace/FlagCX/plugin/torch/flagcx/include/stream_guard_flagcx.hpp"
s = open(P).read()

# 1. 移除 extern 声明（只保留 GetCurrentStream 声明）
old = """#elif USE_ASCEND_ADAPTOR
#include <acl/acl_rt.h>
// torch_fl 暴露的当前 stream 切换 API（csrc/runtime/accelerator/ascend/stream_api.cc）
extern "C" void *GetCurrentStream(int device_index);
extern "C" void SetCurrentStream(int device_index, void *stream);"""
new = """#elif USE_ASCEND_ADAPTOR
#include <acl/acl_rt.h>
// torch_fl 的当前 stream 获取 API（csrc/runtime/accelerator/ascend/stream_api.cc）。
// 说明：collective 已通过 getStreamByIndex(0)=GetCurrentStream(device) 运行在调用者
// 当前 stream 上，无需再切换（libflagos 旧版亦无 SetCurrentStream 导出）。
extern "C" void *GetCurrentStream(int device_index);"""
assert s.count(old) == 1, "extern block not found"
s = s.replace(old, new, 1)

# 2. 构造函数体：ascend 不再 SetCurrentStream
old2 = """  {
#ifdef USE_ASCEND_ADAPTOR
    SetCurrentStream(deviceId_, *(aclrtStream *)stream);
#elif USE_SUNRISE_ADAPTOR
    torchpt::set_current_stream(guard_.unwrap());
#endif
  }"""
new2 = """  {
#ifdef USE_ASCEND_ADAPTOR
    // No-op: collective stream == caller's current stream already.
    (void)stream;
#elif USE_SUNRISE_ADAPTOR
    torchpt::set_current_stream(guard_.unwrap());
#endif
  }"""
assert s.count(old2) == 1, "ctor body not found"
s = s.replace(old2, new2, 1)

# 3. 析构：ascend 不再恢复
old3 = """#ifdef USE_ASCEND_ADAPTOR
  ~flagcxStreamGuard() { SetCurrentStream(deviceId_, prevStream_); }
#elif USE_SUNRISE_ADAPTOR"""
new3 = """#ifdef USE_ASCEND_ADAPTOR
  ~flagcxStreamGuard() = default;
#elif USE_SUNRISE_ADAPTOR"""
assert s.count(old3) == 1, "dtor not found"
s = s.replace(old3, new3, 1)

# 4. reset_stream：ascend 不再 SetCurrentStream
old4 = """#elif USE_ASCEND_ADAPTOR
    guard_ = *(aclrtStream *)stream;
    SetCurrentStream(deviceId_, guard_);"""
new4 = """#elif USE_ASCEND_ADAPTOR
    guard_ = *(aclrtStream *)stream;
    (void)guard_;"""
assert s.count(old4) == 1, "reset_stream not found"
s = s.replace(old4, new4, 1)

open(P, "w").write(s)
print("OK: stream guard ascend 分支移除 SetCurrentStream")
