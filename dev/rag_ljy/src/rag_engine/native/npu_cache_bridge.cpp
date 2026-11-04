// Same public allocator operation as torch.npu.empty_cache(), called via CDLL
// so the Python event loop can continue sampling and rejecting during release.
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <cstdio>
#include <exception>
extern "C" int rag_empty_npu_cache(char* error, size_t capacity) noexcept {
    try {
        c10_npu::NPUCachingAllocator::emptyCache();
        return 0;
    } catch (const std::exception& e) {
        if (capacity) std::snprintf(error, capacity, "%s", e.what());
        return 1;
    } catch (...) {
        if (capacity) std::snprintf(error, capacity, "%s", "Unknown allocator exception");
        return 2;
    }
}
