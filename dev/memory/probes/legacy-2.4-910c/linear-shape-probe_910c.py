#!/usr/bin/env python3
"""大 shape flag_gems linear 复现(vllm qkv 同款规模), 定位是否 shape 相关。"""
import os
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402


def main():
    torch.npu.synchronize()
    from flag_gems.ops.linear import linear

    # vllm Qwen3-4B qkv: (4096, 3*4096), 输入 seq=1..16, hidden=4096
    for M, K, N in [(1, 4096, 12288), (16, 4096, 12288), (4, 4096, 4096)]:
        a = torch.randn(M, K, dtype=torch.bfloat16, device="flagos:0")
        w = torch.randn(N, K, dtype=torch.bfloat16, device="flagos:0")
        t0 = time.time()
        try:
            c = linear(a, w)
            torch.npu.synchronize()
            dt = time.time() - t0
            ref = a.float() @ w.float().T
            err = (c.float() - ref).abs().max().item()
            print(f"linear M={M} K={K} N={N}: {dt*1000:.0f}ms max_err={err:.4f} nans={int((c!=c).sum().item())}", flush=True)
        except Exception as e:
            print(f"linear M={M} K={K} N={N}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        torch.npu.synchronize()

    # 逐层: 单独 torch matmul 同 shape 对照
    a = torch.randn(4, 4096, dtype=torch.bfloat16, device="flagos:0")
    w = torch.randn(12288, 4096, dtype=torch.bfloat16, device="flagos:0")
    t0 = time.time()
    c = a @ w.T
    torch.npu.synchronize()
    print(f"torch matmul M=4 K=4096 N=12288: {(time.time()-t0)*1000:.0f}ms", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
