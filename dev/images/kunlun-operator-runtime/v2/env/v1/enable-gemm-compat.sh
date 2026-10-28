#!/bin/sh
# 昆仑芯 GEMM 兼容修复 · 显式启用（kunlun-operator-runtime v2 · env/v1）
#
# 用法（容器内，跑 GEMM 类算子 mm/bmm/addmm 之前；只影响当前 shell 及其子进程）：
#   source /workspace/dev/images/kunlun-operator-runtime/v2/env/v1/enable-gemm-compat.sh
#
# 做三件事：
#   1. 导入 GEMM compat 必需的四个环境变量（KLX_USE_AUTOTUNE=0 等）；
#   2. 把挂载的补丁版 FlagGems 源树（/workspace/FlagGems/src）与本 compat 目录加入
#      PYTHONPATH 前部 —— 使 import flag_gems 取到补丁版，且 import gemm_compat 可用；
#   3. 提示性核对宿主 FlagGems HEAD（强制核对由 up.sh --strict 完成，此处仅提醒）。
#
# compiler 副本（进程内替换 make_ttxir/make_llir）**不在本脚本完成**：它必须发生在
# 每个 Python 进程内部 —— 你的脚本第一行（任何 torch/flag_gems import 之前）加：
#     import gemm_compat
# 交互验证可用：python -c "import gemm_compat; import flag_gems; print('ok')"
#
# ⚠️ 数值限制（启用即视为已知悉，详见 env/v1 README「数值限制」）：
#   64×64×64 fp32 结果为非确定性乱值（不可信，禁用于任何关键路径）；64×64×64 bf16
#   与大形状 fp16/bf16 的绝对误差超出我组容差口径（口径裁定中）。已验可用面 =
#   minirepro 12 组编译 + FlagGems pytest mm 18 / mm·bmm·addmm 99。

# —— 1. 环境变量（与 2026-10-27 验收形态一致，缺一不可）——
export FLAGTREE_BACKEND=xpu
export KLX_USE_AUTOTUNE=0
export XMLIR_MATMUL_FAST_MODE=1
export XPU_SDNN_PRESERVE_TTXIR_ERRORS=1

# —— 2. import 路径 ——
_SELF_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export PYTHONPATH="/workspace/FlagGems/src${_SELF_DIR}/compat/legacy_sdnn${PYTHONPATH:+:$PYTHONPATH}"

# —— 3. 提示性核对（不阻断）——
if command -v git >/dev/null 2>&1 && [ -e /workspace/FlagGems/.git ]; then
    _head=$(git -C /workspace/FlagGems rev-parse HEAD 2>/dev/null || true)
    if [ "$_head" != "4a00b8451e41ab9b35cff317972d39683121a4dc" ]; then
        echo "⚠️  /workspace/FlagGems HEAD=${_head:-未知} ≠ 补丁分支 4a00b845…" >&2
        echo "    请先在宿主 FlagGems 检出 xliu969/kunlun-gemm-compat 再使用 GEMM 修复。" >&2
    fi
fi
unset _SELF_DIR _head
echo "[gemm-compat] 环境已就绪：FlagGems=补丁版源树；脚本内请先 import gemm_compat 再 import torch/flag_gems。"
