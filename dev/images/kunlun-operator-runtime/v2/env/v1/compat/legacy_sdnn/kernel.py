"""One fused GEMM/Bias/ReLU kernel; no split-kernel/native fallback."""
import triton
import triton.language as tl


@triton.jit
def matmul(a, b, bias, out, m, k, n, BM: tl.constexpr, BN: tl.constexpr,
           BK: tl.constexpr, EPI: tl.constexpr):
    pid = tl.program_id(0)
    columns = tl.cdiv(n, BN)
    rows = (pid // columns) * BM + tl.arange(0, BM)
    cols = (pid % columns) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for base in range(0, k, BK):
        va = tl.load(a + rows[:, None] * k + base + kk[None, :],
                     (rows[:, None] < m) & (base + kk[None, :] < k), 0)
        vb = tl.load(b + (base + kk[:, None]) * n + cols[None, :],
                     (base + kk[:, None] < k) & (cols[None, :] < n), 0)
        acc += tl.dot(va, vb, input_precision='ieee')
    if EPI == 1 or EPI == 3:
        acc += tl.load(bias + cols, cols < n, 0)[None, :]
    if EPI == 2 or EPI == 3:
        acc = tl.maximum(acc, 0)
    tl.store(out + rows[:, None] * n + cols[None, :], acc,
             (rows[:, None] < m) & (cols[None, :] < n))
