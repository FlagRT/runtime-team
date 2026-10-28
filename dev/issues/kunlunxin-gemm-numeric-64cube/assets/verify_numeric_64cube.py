#!/usr/bin/env python3
"""昆仑芯 P800：64×64×64 fp32 GEMM 非确定性乱值 —— 复现 / 修复验收（三态自判定）。

问题：启用 GEMM compat（kunlun-operator-runtime v2 · env/v1）后，FlagGems 路径的
64×64×64 fp32 `torch.mm` 结果为非确定性乱值 —— 同输入多次运行相对误差 0.29~1.02、
无复现规律；同输入厂商原生 `torch.mm` 误差 ~1e-5。该形状不在 FlagGems pytest 的
MNK 矩阵内（pytest 18/99 从未覆盖）。配套现象：64³ bf16 超绝对容差 4.5×；
大形状 fp16/bf16 超本组绝对容差 1.6×~9.8×（相对误差仅 2.8e-4~6e-3，口径之争，
本脚本不判这一部分，数据见 README）。

用法（env/v1 容器内；本脚本头部带自动启用兜底，未 source enable 也可直接跑）：
  CUDA_VISIBLE_DEVICES=<xpu-smi 挑中的空闲卡号> \
    /root/miniconda/envs/python310_torch29_cuda/bin/python3 \
    /workspace/dev/issues/kunlunxin-gemm-numeric-64cube/assets/verify_numeric_64cube.py

判定（CASE-TEMPLATE §2 规则 3，三态）：
  overall: PASS / 0 —— 三次 trial（同固定输入）全部 rel ≤ 1e-3 且结果彼此一致
  overall: FAIL / 1 —— 任一 trial 超阈（数值乱值），或三次结果漂移（非确定性）
  overall: ERROR / 2 —— 环境不满足（无设备 / flag_gems 不可用）

判据口径：相对误差 = 对 FP32 CPU 参考的最大绝对误差 / 参考最大绝对值；fp32 容差
1e-3（与 minirepro / 我组验收口径一致）。
"""
import sys

# 自动启用兜底（env/v1 形态）：已 source enable-gemm-compat.sh 时 sitecustomize
# 已完成全套启用，此处 import 幂等无害；未 source 时按 env/v1 容器内固定路径补位
# （进程内 compiler 副本 + 补丁版 FlagGems 源树 + meta_path 劫持解除）；两者都不可
# 用（非 env/v1 环境，如对方自建容器原始形态）则走原始安装——此时编译失败会被
# 判为 ERROR（修复未启用，无数值可言）。
try:
    import gemm_compat  # noqa: F401
except ImportError:
    _compat_dir = ("/workspace/dev/images/kunlun-operator-runtime/v2/env/v1"
                   "/compat/legacy_sdnn")
    try:
        sys.path.insert(0, _compat_dir)
        import gemm_compat  # noqa: F401
    except Exception:  # noqa: BLE001 —— 非 env/v1 容器：原始安装形态
        pass
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    try:
        import torch
        import flag_gems
    except Exception as e:  # noqa: BLE001
        print(f"overall: ERROR / 2 —— 环境不满足：{e}")
        return 2
    if torch.cuda.device_count() == 0:
        print("overall: ERROR / 2 —— 无可用 XPU 设备（CUDA_VISIBLE_DEVICES 指了空闲卡吗？）")
        return 2

    tol = 1e-3
    trials = 3
    results = []
    compile_error = None
    print(f"flag_gems: {flag_gems.__file__}")
    for t in range(trials):
        try:
            torch.manual_seed(20261028)  # 固定输入：同输入三次运行，非确定性直接可见
            a = torch.randn((64, 64), dtype=torch.float32, device=flag_gems.device)
            b = torch.randn((64, 64), dtype=torch.float32, device=flag_gems.device)
            ref = torch.mm(a.float().cpu(), b.float().cpu())
            native = torch.mm(a, b).float().cpu()  # vendor 原生（不进 use_gems）
            with flag_gems.use_gems():
                out = torch.mm(a, b)
        except BaseException as e:  # noqa: BLE001 —— 编译失败属环境问题（修复未启用），非本 case 判定范围
            compile_error = f"{type(e).__name__}: {str(e)[:200]}"
            break
        out = out.float().cpu()
        d = (out - ref).abs().max().item()
        denom = max(ref.abs().max().item(), 1e-30)
        rel = d / denom
        native_d = (native - ref).abs().max().item()
        results.append(rel)
        print(f"trial{t}: gems rel={rel:.4e} | vendor native max_abs={native_d:.4e}"
              + (f" | out[0,:4]={[round(x, 4) for x in out[0, :4].tolist()]}" if rel > tol else ""))

    if compile_error is not None:
        print(f"overall: ERROR / 2 —— GEMM 编译失败（{compile_error}）——"
              "本 case 须在 env/v1 且显式启用 GEMM 修复后跑（编译失败见 "
              "../kunlunxin-gemm-triton-compile/）；已启用仍失败则撞到另一问题")
        return 2
    over = [r for r in results if r > tol]
    spread = max(results) - min(results)
    if over:
        print(f"判定依据：{len(over)}/{trials} 次 trial rel > {tol}（" +
              ", ".join(f"{r:.3e}" for r in results) + "）")
        print("overall: FAIL / 1 —— 64×64×64 fp32 数值超容差（缺陷复现）")
        return 1
    if spread > tol:
        print(f"判定依据：同输入三次结果漂移（spread={spread:.3e} > {tol}）")
        print("overall: FAIL / 1 —— 非确定性缺陷复现")
        return 1
    print(f"判定依据：三次 trial rel 全部 ≤ {tol}（" +
          ", ".join(f"{r:.3e}" for r in results) + "）")
    print("overall: PASS / 0 —— 64×64×64 fp32 数值在容差内且确定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
