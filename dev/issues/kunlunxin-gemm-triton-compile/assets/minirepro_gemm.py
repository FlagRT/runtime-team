#!/usr/bin/env python3
"""昆仑芯 P800：FlagGems GEMM 在 flagtree 0.6.1+xpu3.6 下编译失败的最小复现 / 修复验收。

一条命令同时给出两件事：
  · 两种失败形态各自是否复现（原生 abort @make_llir / OutOfResources @make_ttxir）；
  · 三态判定（CASE-TEMPLATE §2 规则 3）：
      overall: PASS / 0  —— 全部形状都算对了，缺陷已修复（验收通过）
      overall: FAIL / 1  —— 还有形状编译失败，缺陷复现
      overall: ERROR / 2 —— 环境不满足，下不了判定（无设备、FlagGems 装不上等）

**为什么必须起子进程**：`(1,1,32)` + float16/float32 这几组在 `make_llir` 阶段撞上 LLVM
原生断言，直接 `abort()` 掉整个进程。在本进程里跑会把判定脚本一起带走，拿不到判定行。
每个形状一个子进程，父进程只收结果。

用法（容器内，仓根为 /workspace）：
    CUDA_VISIBLE_DEVICES=<空闲卡号> \
      /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
      /workspace/dev/issues/kunlunxin-gemm-triton-compile/assets/minirepro_gemm.py

可选：`--shapes 1,1,32` 只跑一组；`--child` 是内部用的子进程入口，不要手工调。
"""
import argparse
import json
import os
import subprocess
import sys

# v1 记录的那个 pytest 用例（FlagGems tests/test_blas_ops.py::test_accuracy_mm）在 normal
# 模式下的全部形状，加上 v2 当初手写探针用的 64×64 —— 后者落在另一个失败阶段，一起跑才能
# 看出「形态由 shape × dtype 决定」。形状写死在这里，不依赖镜像内有没有 tests 目录。
SHAPES = [(1, 1, 32), (15, 160, 1024), (495, 5333, 71), (64, 64, 64)]
DTYPES = ["float16", "float32", "bfloat16"]
# 判定用：这两个字符串分别是两种失败形态的特征。abort 形态的 Python 栈顶固定在
# compiler.py:486 make_llir；OutOfResources 形态是 compiler.py:367 对 ttxir 阶段的包装。
SIG_ABORT = "Calling a function with bad signature"
SIG_MAKE_LLIR = "line 486 in make_llir"
SIG_OOR = "OutOfResources: out of resource: uni_sram"


def child(m, n, k, dtype_name):
    """子进程：真的调一次 torch.mm，把结果打成一行 JSON。失败形态由父进程从输出里认。"""
    import torch
    import flag_gems
    dtype = getattr(torch, dtype_name)
    a = torch.randn((m, k), dtype=dtype, device=flag_gems.device)
    b = torch.randn((k, n), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(a.float().cpu(), b.float().cpu())
    try:
        with flag_gems.use_gems():
            out = torch.mm(a, b)
    except BaseException as e:                      # noqa: BLE001 —— 要连 SystemExit 一起收
        print("CHILD " + json.dumps({"status": "raised", "exc": type(e).__name__,
                                     "msg": str(e)[:300]}))
        return 0
    # 算出来了还要对数值：编译通过但结果错同样不能算修复
    diff = (out.float().cpu() - ref).abs().max().item()
    tol = {"float16": 1e-2, "bfloat16": 5e-2, "float32": 1e-3}[dtype_name]
    print("CHILD " + json.dumps({"status": "ok", "max_abs_diff": diff,
                                 "tol": tol, "within_tol": diff <= tol}))
    return 0


def run_one(m, n, k, dtype_name):
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child",
         "--shape", f"{m},{n},{k}", "--dtype", dtype_name],
        capture_output=True, text=True, timeout=1800)
    text = proc.stdout + proc.stderr
    payload = {}
    for line in proc.stdout.splitlines():
        if line.startswith("CHILD "):
            payload = json.loads(line[len("CHILD "):])
    form, verdict = "unknown", "unknown"
    if SIG_ABORT in text or SIG_MAKE_LLIR in text:
        form = "native-abort@make_llir"
        verdict = "repro"
    elif SIG_OOR in text or (payload.get("exc") == "OutOfResources"):
        form = "OutOfResources:uni_sram@make_ttxir"
        verdict = "repro"
    elif payload.get("status") == "ok":
        form = "compiled"
        verdict = "fixed" if payload.get("within_tol") else "wrong-numerics"
    elif payload.get("status") == "raised":
        form = f"other-exception:{payload.get('exc')}"
        verdict = "inconclusive"
    return {"shape": [m, n, k], "dtype": dtype_name, "rc": proc.returncode,
            "form": form, "verdict": verdict, "detail": payload,
            "abort_marker": SIG_ABORT in text, "make_llir_marker": SIG_MAKE_LLIR in text}


def preflight():
    """环境自检：不满足就是 ERROR（无法下判定），不要误报成 FAIL（缺陷复现）。"""
    probe = ("import torch, flag_gems; "
             "print('PRE', torch.cuda.device_count(), flag_gems.device)")
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                          timeout=900)
    line = next((x for x in proc.stdout.splitlines() if x.startswith("PRE ")), None)
    if proc.returncode != 0 or not line:
        return False, (proc.stderr.strip() or proc.stdout.strip())[-400:]
    count = line.split()[1]
    if count == "0":
        return False, ("torch.cuda.device_count() == 0 —— 检查 /dev/xpuctrl 是否授予，以及 "
                       "CUDA_VISIBLE_DEVICES 是否按容器内重编号后的逻辑号写"
                       "（见 dev/images/kunlun-operator-runtime/v2/demo/docker-compose.yml 文件头）")
    return True, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shape")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--shapes", help="只跑这些形状，形如 1,1,32;64,64,64")
    args = ap.parse_args()
    if args.child:
        m, n, k = (int(x) for x in args.shape.split(","))
        return child(m, n, k, args.dtype)

    print("=== minirepro_gemm: FlagGems GEMM 编译失败（昆仑芯 P800 / flagtree 0.6.1+xpu3.6）===")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(未设)')}")
    ok, info = preflight()
    print(f"preflight: {info}")
    if not ok:
        print("\n=== VERDICT (minirepro_gemm) ===")
        print("  环境不满足，下不了判定")
        print("overall: ERROR")
        return 2

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    results = [run_one(*s, d) for s in shapes for d in DTYPES]

    print("\n--- 逐组形态 ---")
    print(f"{'M-N-K':<16}{'dtype':<10}{'rc':<4}{'form'}")
    for r in results:
        print(f"{'-'.join(map(str, r['shape'])):<16}{r['dtype']:<10}{r['rc']:<4}{r['form']}")

    abort_n = sum(1 for r in results if r["form"] == "native-abort@make_llir")
    oor_n = sum(1 for r in results if r["form"].startswith("OutOfResources"))
    fixed_n = sum(1 for r in results if r["verdict"] == "fixed")
    bad_num = [r for r in results if r["verdict"] == "wrong-numerics"]
    unclear = [r for r in results if r["verdict"] in ("inconclusive", "unknown")]

    print("\n=== VERDICT (minirepro_gemm) ===")
    print(f"  组数: {len(results)}  原生 abort: {abort_n}  OutOfResources: {oor_n}  "
          f"编译并算对: {fixed_n}")
    if abort_n:
        print(f"  repro: CONFIRMED - {abort_n} 组在 make_llir 阶段原生 abort"
              f"（LLVM 断言 \"Calling a function with bad signature!\"）")
    if oor_n:
        print(f"  repro: CONFIRMED - {oor_n} 组在 ttxir 阶段失败，被 compiler.py:367 误标为 "
              f"OutOfResources: uni_sram（Required/Hardware limit 是硬编码零，并非资源不足）")
    if bad_num:
        print(f"  编译通过但数值错 {len(bad_num)} 组 —— 不算修复: "
              f"{[r['shape'] + [r['dtype']] for r in bad_num]}")
    if unclear:
        print(f"  无法归类 {len(unclear)} 组，下不了判定: "
              f"{[(r['shape'], r['dtype'], r['form']) for r in unclear]}")
        print("overall: ERROR")
        return 2
    if fixed_n == len(results):
        print("  缺陷已修复：全部形状编译通过且数值在容差内")
        print("overall: PASS")
        return 0
    print("overall: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
