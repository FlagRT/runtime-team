#!/usr/bin/env python3
"""最小 triton kernel 冒烟:测 ascend launch 路径本身(不经过 vllm)。

一个需要真实计算的小 kernel + 连续多次 launch,若 triton_ascend launch 链路健康应秒回。
"""
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _add_kernel(x_ptr, y_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(z_ptr + offs, x + y, mask=mask)


def main():
    torch.npu.synchronize()
    n = 4096
    BLOCK = 1024
    x = torch.randn(n, dtype=torch.float32, device="flagos:0")
    y = torch.randn(n, dtype=torch.float32, device="flagos:0")
    z = torch.empty(n, dtype=torch.float32, device="flagos:0")
    grid = (n // BLOCK,)
    for i in range(5):
        t0 = time.time()
        _add_kernel[grid](x, y, z, n, BLOCK_SIZE=BLOCK)
        torch.npu.synchronize()
        dt = time.time() - t0
        print(f"iter {i}: {dt*1000:.1f} ms  ok={bool(torch.allclose(z, x + y))}", flush=True)
    print("TRITON SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
