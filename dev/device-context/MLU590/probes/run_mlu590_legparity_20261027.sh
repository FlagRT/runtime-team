#!/bin/bash
# =============================================================================
# 寒武纪 MLU590 · `DC_MODEL` 两腿口径复验（2026-10-27）—— 共享层改动：训练腿补 resolve_model
#
# 缺陷的**发现地**就是本实例：10-27 复跑时同一份
#   `DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B`
#   推理腿过、训练腿报 `ValueError: Unrecognized model in …/models--Qwen--Qwen3-Embedding-0.6B`
#   （当时在容器内先手工把 DC_MODEL 解析成 snapshots/<hash> 绕过，没在回归记录里夹带改共享层）
# 修法：抽成 `runtime/proto/model_path.py` 两腿共用 + `scripts/model_path_parity_check.py` 静态同源守卫。
#
# 本文件只是**薄封装**：判据本体在 `prototype/scripts/run_legparity_dcmodel.sh`（四实例共用一份）。
#
# 环境事实（沿用 10-27 复跑的实测约定）：
#   容器 dc-mlu590-hliu553（自 2026-09-22 Up，**长期复用容器**）· 容器内 /work == 宿主 /srv/hliu553
#   解释器 /flagos/bin/python3（py3.10 / torch 2.7.1+cpu / torch_mlu 1.29.2）
#   选卡 MLU_VISIBLE_DEVICES（容器内 0 基）；⚠️ **DC_DIST_BT 必须显式 `cncl`**
#     （本原型对寒武纪拒绝兜底 gloo：那会静默退化成纯 CPU 集合通信，训练照样跑完但设备侧通信没被验证）
#   模型 snapshot = /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
#
# 用法（**宿主上跑**，脚本自己 docker exec）：
#   bash run_mlu590_legparity_20261027.sh
# =============================================================================
set -u
export C=${C:-dc-mlu590-hliu553}
export PY=${PY:-/flagos/bin/python3}
export PROTO=${PROTO:-/work/dc_rerun_20261027/dev/device-context/prototype}
export WORK=${WORK:-/work/dc_rerun_20261027}
# ⚠️ 宿主/容器路径**必须分开给**（这条坑本方向已踩过 9 次以上）：日志在宿主落盘，容器里跑的是另一视角
export HOST_WORK=${HOST_WORK:-/srv/hliu553/dc_rerun_20261027}
export MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
export VIS_VAR=${VIS_VAR:-MLU_VISIBLE_DEVICES}
export DC_BACKEND=${DC_BACKEND:-cambricon}
export DC_DIST_BT=${DC_DIST_BT:-cncl}
export DEVS=${DEVS:-0,1}
export MAX_STEPS=${MAX_STEPS:-50}

CORE=${CORE:-$(cd "$(dirname "$0")/../../prototype/scripts" 2>/dev/null && pwd)/run_legparity_dcmodel.sh}
[ -f "$CORE" ] || { echo "找不到判据核心脚本：$CORE（用 CORE=<路径> 显式给）" >&2; exit 2; }
exec bash "$CORE"
