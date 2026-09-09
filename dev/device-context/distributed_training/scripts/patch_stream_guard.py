#!/usr/bin/env python3
"""stream_guard_flagcx.hpp: ascend 分支真正切换 torch_fl 当前 stream"""
P = "/workspace/FlagCX/plugin/torch/flagcx/include/stream_guard_flagcx.hpp"
s = open(P).read()

# 1. ascend include 区加 extern 声明
old = """#elif USE_ASCEND_ADAPTOR
#include <acl/acl_rt.h>"""
new = """#elif USE_ASCEND_ADAPTOR
#include <acl/acl_rt.h>
// torch_fl 暴露的当前 stream 切换 API（csrc/runtime/accelerator/ascend/stream_api.cc）
extern "C" void *GetCurrentStream(int device_index);
extern "C" void SetCurrentStream(int device_index, void *stream);"""
assert s.count(old) == 1, "ascend include block not found"
s = s.replace(old, new, 1)

# 2. 构造函数：保存旧流 + 切换
old2 = """      : originalStream_(stream), currentStream_(nullptr), deviceId_(deviceId),
#ifdef USE_NVIDIA_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))"""
new2 = """      : originalStream_(stream), currentStream_(nullptr), deviceId_(deviceId),
#ifdef USE_ASCEND_ADAPTOR
        prevStream_(GetCurrentStream(deviceId))
#elif USE_NVIDIA_ADAPTOR
        guard_(
            at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId))"""
assert s.count(old2) == 1, "ctor head not found"
s = s.replace(old2, new2, 1)

# 3. 构造函数体：ascend 分支切换当前流
old3 = """#elif USE_ASCEND_ADAPTOR
        guard_(*(aclrtStream *)stream)
#elif USE_AMD_ADAPTOR"""
new3 = """#elif USE_ASCEND_ADAPTOR
        guard_(*(aclrtStream *)stream)
#elif USE_AMD_ADAPTOR"""
assert s.count(old3) == 1, "ascend guard init not found"
s = s.replace(old3, new3, 1)

# 4. 构造体（{} 内）ascend 切换
old4 = """  {
#ifdef USE_SUNRISE_ADAPTOR
    torchpt::set_current_stream(guard_.unwrap());
#endif
  }"""
new4 = """  {
#ifdef USE_ASCEND_ADAPTOR
    SetCurrentStream(deviceId_, *(aclrtStream *)stream);
#elif USE_SUNRISE_ADAPTOR
    torchpt::set_current_stream(guard_.unwrap());
#endif
  }"""
assert s.count(old4) == 1, "ctor body not found"
s = s.replace(old4, new4, 1)

# 5. 析构：ascend 恢复旧流
old5 = """#ifdef USE_SUNRISE_ADAPTOR
  // torchpt::PTPUStream is a value type, not an RAII guard, so we have
  // to restore the previous current stream by hand on destruction.
  ~flagcxStreamGuard() {
    torchpt::set_current_stream(sunrisePrevStream_.unwrap());
  }
#else
  ~flagcxStreamGuard() = default;
#endif"""
new5 = """#ifdef USE_ASCEND_ADAPTOR
  ~flagcxStreamGuard() { SetCurrentStream(deviceId_, prevStream_); }
#elif USE_SUNRISE_ADAPTOR
  // torchpt::PTPUStream is a value type, not an RAII guard, so we have
  // to restore the previous current stream by hand on destruction.
  ~flagcxStreamGuard() {
    torchpt::set_current_stream(sunrisePrevStream_.unwrap());
  }
#else
  ~flagcxStreamGuard() = default;
#endif"""
assert s.count(old5) == 1, "dtor not found"
s = s.replace(old5, new5, 1)

# 6. reset_stream：ascend 分支切换
old6 = """#elif USE_ASCEND_ADAPTOR
    guard_ = *(aclrtStream *)stream;"""
new6 = """#elif USE_ASCEND_ADAPTOR
    guard_ = *(aclrtStream *)stream;
    SetCurrentStream(deviceId_, guard_);"""
assert s.count(old6) == 1, "reset_stream not found"
s = s.replace(old6, new6, 1)

# 7. 成员：ascend 加 prevStream_
old7 = """#elif USE_ASCEND_ADAPTOR
  aclrtStream guard_;
#elif USE_AMD_ADAPTOR"""
new7 = """#elif USE_ASCEND_ADAPTOR
  aclrtStream guard_;
  void *prevStream_;
#elif USE_AMD_ADAPTOR"""
assert s.count(old7) == 1, "members not found"
s = s.replace(old7, new7, 1)

open(P, "w").write(s)
print("OK: flagcxStreamGuard ascend 分支已实现真正切换")
