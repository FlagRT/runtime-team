#!/usr/bin/env python3
"""shmem 缺失：最小复现 / 修复验收（1 卡，自判定，不需要模型权重）。

    python3 minirepro_shmem.py

编译并执行一个唯一特征是「含一个 for 循环」的 Triton kernel。ascend 后端编译器前端
（triton/spec/ascend/compiler/code_generator.py:1177 visit_For，第 1197 行）对任何 for 循环都
import triton.experimental.tle，该模块顶层经 experimental/tle/backends.py:12 _build_ascend_impl
无条件 import shmem（experimental/tle/language/dsa/ascend/communication.py:6），
而 shmem 不在镜像内。因此本 kernel 的编译即命中缺陷，与用哪个算子、是否加载模型无关。

判定（统一约定见 dev/issues/README.md）：
  overall: PASS   退出码 0   kernel 编译通过且数值正确 -> 本环境不存在该缺陷
  overall: FAIL   退出码 1   编译失败于 No module named 'shmem' -> 缺陷复现（repro: CONFIRMED）
  overall: ERROR  退出码 2   其他失败或数值不符 -> 另一个问题，本脚本不下判定

实现注记：
- 每次运行使用一个全新的 Triton 编译缓存目录。缺省缓存目录 ~/.triton/cache 在容器内跨运行
  累积（镜像本身不含编译缓存，/root/.triton 下是 llvm 等构建依赖）：同一容器内若先跑过一次
  带 stub 的成功编译，再撤除 stub 重跑会直接命中已编译产物、不再走 visit_For，缺陷因此不
  复现、判定假 PASS。实测遇到过。
- 读入与写出使用两个不同张量。原地读改写（in/out 同一张量）在本后端得到错误数值，
  与 shmem 缺陷无关，会把判定污染成 ERROR。
"""
import os
import sys
import tempfile
import traceback

# 必须在 import triton 之前设置。
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="minirepro-triton-cache-")

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

N = 8192
BLOCK = 128
SUB = 32


@triton.jit
def loop_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr, SUB: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    # 这个 for 循环是唯一的触发点：visit_For -> import triton.experimental.tle -> import shmem
    for i in range(0, BLOCK, SUB):
        offs = base + i + tl.arange(0, SUB)
        m = offs < n
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m) * 2.0, mask=m)


def verdict(line, detail, code):
    print("\n=== VERDICT (minirepro_shmem, 1 card) ===", flush=True)
    print(f"  {detail}", flush=True)
    print(f"overall: {line}", flush=True)
    sys.stdout.flush()
    sys.exit(code)


def main():
    print(f"TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']} (fresh)", flush=True)
    x = torch.arange(N, dtype=torch.float32, device="npu")
    y = torch.zeros(N, dtype=torch.float32, device="npu")
    try:
        loop_kernel[(triton.cdiv(N, BLOCK),)](x, y, N, BLOCK=BLOCK, SUB=SUB)
        torch_npu.npu.synchronize()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        if "No module named 'shmem'" in tb:
            verdict("FAIL", "repro: CONFIRMED - for-loop kernel compilation hit "
                            "ModuleNotFoundError: No module named 'shmem'", 1)
        verdict("ERROR", "compilation/execution failed for another reason "
                         "(see traceback above), not this defect", 2)

    got, ref = y.cpu(), torch.arange(N, dtype=torch.float32) * 2.0
    ok = torch.allclose(got, ref)
    print(f"MINIREPRO_OK={ok} y[:5]={got[:5].tolist()}", flush=True)
    if not ok:
        verdict("ERROR", "kernel compiled but the numeric result is wrong; "
                         "a different problem, not shmem", 2)
    verdict("PASS", "for-loop kernel compiled and the numeric result is correct "
                    "-> no shmem gap in this environment", 0)


if __name__ == "__main__":
    main()
