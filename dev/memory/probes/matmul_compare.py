#!/usr/bin/env python3
"""对比: torch 原生 matmul vs flag_gems linear, 确认 AIC 路径哪个坏了。"""
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import torch  # noqa: E402


def main():
    torch.npu.synchronize()
    a = torch.randn(256, 512, dtype=torch.float16, device="flagos:0")
    b = torch.randn(512, 128, dtype=torch.float16, device="flagos:0")

    # 1) torch 原生 matmul(aclnn)
    c1 = a @ b
    torch.npu.synchronize()
    ref = (a.float() @ b.float())
    err1 = (c1.float() - ref).abs().max().item()
    nan1 = int((c1 != c1).sum().item())
    print(f"torch native matmul: max_abs_err={err1:.6f}  nans={nan1}", flush=True)

    # 2) flag_gems linear
    try:
        from flag_gems.ops.linear import linear
        w = b.clone()  # (out, in)
        c2 = linear(a, w)
        torch.npu.synchronize()
        err2 = (c2.float() - ref).abs().max().item()
        nan2 = int((c2 != c2).sum().item())
        print(f"flag_gems linear:    max_abs_err={err2:.6f}  nans={nan2}", flush=True)
    except Exception as e:
        print(f"flag_gems linear failed: {type(e).__name__}: {str(e)[:150]}", flush=True)

    # 3) flag_gems matmul op (经 dispatch)
    try:
        import flag_gems
        c3 = flag_gems.matmul(a, b)
        torch.npu.synchronize()
        err3 = (c3.float() - ref).abs().max().item()
        nan3 = int((c3 != c3).sum().item())
        print(f"flag_gems matmul:    max_abs_err={err3:.6f}  nans={nan3}", flush=True)
    except Exception as e:
        print(f"flag_gems matmul failed: {type(e).__name__}: {str(e)[:150]}", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
