#!/usr/bin/env python3
"""量化 mm 误差 + 检查 AIC kernel 结果错误模式。"""
import os

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
    _mm_kernel[grid](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    torch.npu.synchronize()
    ref = (a.float() @ b.float())
    diff = (c.float() - ref).abs()
    print(f"max_abs_err={diff.max().item():.6f}  mean_abs_err={diff.mean().item():.6f}")
    print(f"ref sample: {ref[0,:4].tolist()}")
    print(f"c   sample: {c[0,:4].tolist()}")
    # 看看 c 是不是全 0 / 全 1 之类
    print(f"c nonzero frac: {(c != 0).float().mean().item():.4f}  c std: {c.float().std().item():.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
