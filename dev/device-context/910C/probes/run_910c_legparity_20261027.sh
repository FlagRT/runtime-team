#!/bin/bash
# =============================================================================
# 910C · `DC_MODEL` 两腿口径复验（2026-10-27）—— 共享层改动：训练腿补 resolve_model
#
# 缺陷：`proto_infer_leg` 有 `resolve_model()`（HF 缓存根 → snapshots/<hash>），
#       `proto_train_leg` 没有 ⇒ 同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--…`
#       推理腿过、训练腿报 `ValueError: Unrecognized model in …`（10-27 MLU590 实测暴露）。
# 修法：抽成 `runtime/proto/model_path.py` 两腿共用 + `scripts/model_path_parity_check.py` 静态同源守卫。
#
# 本文件只是**薄封装**：判据本体在 `prototype/scripts/run_legparity_dcmodel.sh`（四实例共用一份，
# 避免四个实例各写一套判据又慢慢漂移 —— 与本轮修的正是同一类问题）。
#
# 环境事实（沿用既有实测约定，不另立）：
#   容器 dc-rerun-1027-910c（**只挂 davinci6/7**）· 容器内 /workspace == 宿主 /home/hliu553/dc-rerun-1027
#   解释器 python3（镜像 flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64）
#   选卡 ASCEND_RT_VISIBLE_DEVICES（**容器内 0 基**；填错 ⇒ `aclInit 107001`，像硬件坏）
#   模型 /workspace/models/Qwen3-Embedding-0.6B（本实例无 HF 缓存形态 ⇒ 合成缓存由核心脚本造）
#
# 用法（**宿主上跑**，脚本自己 docker exec）：
#   bash run_910c_legparity_20261027.sh
# =============================================================================
set -u
export C=${C:-dc-rerun-1027-910c}
export PY=${PY:-python3}
export PROTO=${PROTO:-/workspace/dev/device-context/prototype}
export WORK=${WORK:-/workspace}
# ⚠️ 宿主/容器路径**必须分开给**（这条坑本方向已踩过 9 次以上）：日志在宿主落盘，容器里跑的是另一视角
export HOST_WORK=${HOST_WORK:-/home/hliu553/dc-rerun-1027}
export MODEL=${MODEL:-/workspace/models/Qwen3-Embedding-0.6B}
export VIS_VAR=${VIS_VAR:-ASCEND_RT_VISIBLE_DEVICES}
export DC_BACKEND=${DC_BACKEND:-ascend}
export DC_DIST_BT=${DC_DIST_BT:-hccl}
export DEVS=${DEVS:-0,1}
export MAX_STEPS=${MAX_STEPS:-50}

CORE=${CORE:-$(cd "$(dirname "$0")/../../prototype/scripts" 2>/dev/null && pwd)/run_legparity_dcmodel.sh}
[ -f "$CORE" ] || { echo "找不到判据核心脚本：$CORE（用 CORE=<路径> 显式给）" >&2; exit 2; }
exec bash "$CORE"
