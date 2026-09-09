#!/usr/bin/env python3
"""同一 shape 调两次: 区分 triton 编译开销 vs kernel 真慢。"""
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402


def main():
    torch.npu.synchronize()
    from flag_gems.ops.linear import linear

    for M in (1, 16):
        K, N = 4096, 12288
        a = torch.randn(M, K, dtype=torch.bfloat16, device="flagos:0")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="flagos:0")
        for i in range(3):
            t0 = time.time()
            c = linear(a, w)
            torch.npu.synchronize()
            dt = time.time() - t0
            err = (c.float() - a.float() @ w.float().T).abs().max().item()
            print(f"M={M} call{i}: {dt*1000:.0f}ms err={err:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
