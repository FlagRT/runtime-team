// c10_npu shim for the flagcx torch plugin (torch_fl route, no torch_npu).
//
// 提供 flagcx._C 需要的 8 个 c10_npu 符号(NPUEvent/NPUStream/getCurrentNPUStream),
// 用 CANN ACL 直接实现,避免加载真实 libtorch_npu.so:
// 真实库的静态初始化会注册 PrivateUse1 backend fallback,与 torch_fl 冲突
// (实测 terminate: "Tried to register multiple backend fallbacks for the same
// dispatch key PrivateUse1")。
//
// 编译(容器内):
//   g++ -std=c++17 -O2 -shared -fPIC c10_npu_shim.cpp -o libtorch_npu.so \
//     -I/root/vllm-venv312/lib/python3.12/site-packages/torch_npu/include \
//     -I/root/vllm-venv312/lib/python3.12/site-packages/torch/include \
//     -I/root/vllm-venv312/lib/python3.12/site-packages/torch/include/torch/csrc/api/include \
//     -I/usr/local/Ascend/cann-9.0.0/aarch64-linux/include \
//     -L/usr/local/Ascend/cann-9.0.0/lib64 -lascendcl
// 产出替换 site-packages/torch_npu/lib/libtorch_npu.so(构建期链接 + 运行期 dlopen 共用)。
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"

#include <acl/acl_rt.h>
#include <c10/core/Device.h>
#include <c10/core/Stream.h>

#include <cstdint>
#include <map>
#include <mutex>

namespace c10_npu {

namespace {

std::mutex g_stream_mu;
std::map<int, aclrtStream> g_streams;

aclrtStream raw_stream_for(int device) {
    std::lock_guard<std::mutex> lock(g_stream_mu);
    auto it = g_streams.find(device);
    if (it != g_streams.end()) {
        return it->second;
    }
    aclrtStream s = nullptr;
    if (aclrtCreateStream(&s) != ACL_SUCCESS) {
        return nullptr;
    }
    g_streams[device] = s;
    return s;
}

NPUStream npu_stream_for(int device) {
    aclrtStream raw = raw_stream_for(device);
    // aclrtStream 是 64 位指针,塞进 c10::StreamId,stream() 时取回。
    c10::Stream cs(c10::Stream::UNSAFE,
                   c10::Device(c10::DeviceType::PrivateUse1, device),
                   reinterpret_cast<c10::StreamId>(raw));
    return NPUStream(NPUStream::UNCHECKED, cs);
}

}  // namespace

NPUStream getCurrentNPUStream(c10::DeviceIndex device) {
    return npu_stream_for(device);
}

NPUStream getNPUStreamFromPool(c10::DeviceIndex device) {
    return npu_stream_for(device);
}

aclrtStream NPUStream::stream(const bool /*need_empty*/) const {
    return reinterpret_cast<aclrtStream>(stream_.id());
}

NPUEvent::NPUEvent() : event_(nullptr) {
    aclrtCreateEvent(&event_);
}

NPUEvent::~NPUEvent() {
    if (event_ != nullptr) {
        aclrtDestroyEvent(event_);
        event_ = nullptr;
    }
}

NPUEvent& NPUEvent::operator=(NPUEvent&& other) {
    if (this != &other) {
        if (event_ != nullptr) {
            aclrtDestroyEvent(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void NPUEvent::record(const NPUStream& stream) {
    aclrtRecordEvent(event_, stream.stream(false));
}

void NPUEvent::block(const NPUStream& stream) {
    aclrtStreamWaitEvent(stream.stream(false), event_);
}

}  // namespace c10_npu
