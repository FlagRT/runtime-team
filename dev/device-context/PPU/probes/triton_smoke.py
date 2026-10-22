#!/usr/bin/env python3
"""PPU triton 非空转验证：能 import ≠ 能用。

编一个真实 triton kernel（向量加），在 PPU 上执行并核对数值：
  - 走完整链路 triton.compile（PPU 后端）→ launch → 结果比对
  - 只有「编译通过 + 数值正确」才算 triton 真的可用
用法：docker exec <容器> python3 -u /workspace/probes_ppu/triton_smoke.py
"""
import sys
import time

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def main():
    print(f"triton {triton.__version__} | torch {torch.__version__}")
    print(f"triton file = {triton.__file__}")
    print(f"device_count = {torch.cuda.device_count()} | dev0 = {torch.cuda.get_device_name(0)}")

    n = 4096
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    ref = x + y

    t0 = time.time()
    grid = (triton.cdiv(n, 1024),)
    _add_kernel[grid](x, y, out, n, BLOCK=1024)
    torch.cuda.synchronize()
    dt = time.time() - t0

    ok = torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    maxdiff = (out - ref).abs().max().item()
    print(f"[编译+执行] 耗时 {dt:.2f}s  最大误差 {maxdiff:.3e}  一致={ok}")

    # 第二组：非整块尺寸（走 mask 边界）
    n2 = 3000
    a = torch.arange(n2, device="cuda", dtype=torch.float32)
    b = torch.ones(n2, device="cuda", dtype=torch.float32)
    o2 = torch.empty_like(a)
    _add_kernel[(triton.cdiv(n2, 1024),)](a, b, o2, n2, BLOCK=1024)
    torch.cuda.synchronize()
    ok2 = torch.allclose(o2, a + b, atol=1e-5)
    print(f"[边界尺寸 n={n2}] 一致={ok2}")

    verdict = "TRITON_PPU_SMOKE_PASS" if (ok and ok2) else "TRITON_PPU_SMOKE_FAIL"
    print(verdict)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
