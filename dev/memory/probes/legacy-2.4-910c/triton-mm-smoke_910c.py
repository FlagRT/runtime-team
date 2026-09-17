#!/usr/bin/env python3
"""最小 matmul(dot)kernel:验证 AIC cube 路径在 flagos + triton_ascend 3.2.2 是否可用。

若此 kernel 也卡死 → 问题在 triton_ascend AIC 路径本身(与 vllm/attention/workspace 无关)。
"""
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _mm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] + offs_n[None, :] * K
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BK
        b_ptrs += BK * K
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc)


def main():
    torch.npu.synchronize()
    M, N, K = 128, 128, 128
    BM = BN = BK = 64
    a = torch.randn(M, K, dtype=torch.float16, device="flagos:0")
    b = torch.randn(K, N, dtype=torch.float16, device="flagos:0")
    c = torch.empty(M, N, dtype=torch.float32, device="flagos:0")
    grid = (M // BM, N // BN)
    for i in range(3):
        t0 = time.time()
        _mm_kernel[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
        torch.npu.synchronize()
        dt = time.time() - t0
        ref = (a.float() @ b.float())
        ok = bool(torch.allclose(c, ref, atol=0.05, rtol=0.05))
        print(f"mm iter {i}: {dt*1000:.1f} ms ok={ok}", flush=True)
    print("MM SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
