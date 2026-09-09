#!/usr/bin/env python3
"""FlagCX plugin/torch Ascend adaptation: replace torch_npu (NPUEvent/NPUStream)
dependency with CANN ACL APIs (aclrtEvent / aclrtStream), so the comm layer
(FlagCX) consumes the vendor runtime directly instead of torch_npu.
This aligns with the FlagOS/torch_fl ecosystem (no torch_npu)."""
import sys

FAILED = []

def patch(path, old, new, label):
    try:
        s = open(path).read()
    except Exception as e:
        print(f"[{label}] READ FAIL: {e}")
        FAILED.append(label)
        return
    if old not in s:
        print(f"[{label}] PATTERN NOT FOUND, abort for this file")
        FAILED.append(label)
        return
    s = s.replace(old, new)
    open(path, "w").write(s)
    print(f"[{label}] OK")

BASE = "/workspace/FlagCX/plugin/torch/flagcx"

# ---------- 1. event_flagcx.hpp: include ----------
patch(f"{BASE}/include/event_flagcx.hpp",
'''#elif USE_ASCEND_ADAPTOR
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"''',
'''#elif USE_ASCEND_ADAPTOR
#include <acl/acl_rt.h>''',
"event_flagcx.hpp include")

# ---------- 2. event_flagcx.hpp: flagcxCannEvent class ----------
patch(f"{BASE}/include/event_flagcx.hpp",
'''#elif USE_ASCEND_ADAPTOR
class flagcxCannEvent : public flagcxEvent {
public:
  flagcxCannEvent() { npu_event = c10_npu::NPUEvent(); }

  void record(const int device_id) override {
    npu_event.record(c10_npu::getCurrentNPUStream(device_id));
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    npu_event.record(c10_npu::getNPUStreamFromPool(device_id));
  }

  void block(const int device_id) override {
    npu_event.block(c10_npu::getCurrentNPUStream(device_id));
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    npu_event.block(c10_npu::getNPUStreamFromPool(device_id));
  }

private:
  c10_npu::NPUEvent npu_event;
};''',
'''#elif USE_ASCEND_ADAPTOR
class flagcxCannEvent : public flagcxEvent {
public:
  flagcxCannEvent() { aclrtCreateEvent(&event_); }
  ~flagcxCannEvent() override {
    if (event_) aclrtDestroyEvent(event_);
  }

  void record(const int device_id) override {
    aclrtStream st = nullptr;
    aclrtGetCurrentStream(&st);
    aclrtRecordEvent(event_, st);
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    aclrtRecordEvent(event_, *(aclrtStream *)stream);
  }

  void block(const int device_id) override {
    aclrtStream st = nullptr;
    aclrtGetCurrentStream(&st);
    aclrtStreamWaitEvent(st, event_);
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    aclrtStreamWaitEvent(*(aclrtStream *)stream, event_);
  }

private:
  aclrtEvent event_ = nullptr;
};''',
"event_flagcx.hpp CannEvent class")

# ---------- 3. stream_guard_flagcx.hpp: include ----------
patch(f"{BASE}/include/stream_guard_flagcx.hpp",
'''#elif USE_ASCEND_ADAPTOR
#include "torch_npu/csrc/core/npu/NPUStream.h"''',
'''#elif USE_ASCEND_ADAPTOR
#include <acl/acl_rt.h>''',
"stream_guard_flagcx.hpp include")

# ---------- 4. backend_flagcx.cpp: getStreamByIndex ----------
patch(f"{BASE}/src/backend_flagcx.cpp",
'''#ifdef USE_ASCEND_ADAPTOR
    // TODO: The getStreamFromExternal interface is not supported at this stage
    // on NPU. Adaptation modifications will be made in the future.
    acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    flagcxStreams_[streamId] = reinterpret_cast<flagcxStream_t>(&acl_stream);
#else
    C10D_FLAGCX_CHECK(devHandle_->streamCreate(&flagcxStreams_[streamId]),
                      std::nullopt);
#endif''',
'''    // FlagOS adaptation: use flagcx device-handle stream creation (ACL-based),
    // replacing torch_npu's c10_npu stream on Ascend.
    C10D_FLAGCX_CHECK(devHandle_->streamCreate(&flagcxStreams_[streamId]),
                      std::nullopt);''',
"backend_flagcx.cpp getStreamByIndex")

if FAILED:
    print(f"\nPATCH INCOMPLETE, failed: {FAILED}")
    sys.exit(1)
print("\nALL PATCHES APPLIED")
