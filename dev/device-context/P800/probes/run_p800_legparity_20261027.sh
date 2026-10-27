#!/bin/bash
# =============================================================================
# 昆仑芯 P800 · `DC_MODEL` 两腿口径复验（2026-10-27）—— 共享层改动：训练腿补 resolve_model
#
# 缺陷：`proto_infer_leg` 有 `resolve_model()`（HF 缓存根 → snapshots/<hash>），
#       `proto_train_leg` 没有 ⇒ 同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--…`
#       推理腿过、训练腿报 `ValueError: Unrecognized model in …`（10-27 MLU590 实测暴露）。
# 修法：抽成 `runtime/proto/model_path.py` 两腿共用 + `scripts/model_path_parity_check.py` 静态同源守卫。
#
# 本文件只是**薄封装**：判据本体在 `prototype/scripts/run_legparity_dcmodel.sh`（四实例共用一份）。
#
# 环境事实（沿用 10-26 复跑的实测约定）：
#   容器 hliu553-device-context-p800（10-27 11:39 由 `docker start` 拉起；此前为 Exited）
#   容器内 /workspace == 宿主 /data2/hliu553
#   ⚠️ 解释器**不是**默认 python3（/root/miniconda/bin/python3 里没有 torch）：
#      用 /root/miniconda/envs/python310_torch29_cuda/bin/python3（torch 2.9.0+cu129）
#   选卡 CUDA_VISIBLE_DEVICES（容器内 0 基）；DC_DIST_BT="cpu:gloo,cuda:flagcx"（flagcx 需显式 import）
#   模型 /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
#   ⛔ 不用卡 1（10-26 已登记为**已知故障卡**：0 MiB 但不干活）
#
# 用法（**宿主上跑**，脚本自己 docker exec）：
#   bash run_p800_legparity_20261027.sh
# =============================================================================
set -u
export C=${C:-hliu553-device-context-p800}
export PY=${PY:-/root/miniconda/envs/python310_torch29_cuda/bin/python3}
export PROTO=${PROTO:-/workspace/dc_rerun_20261026/prototype}
export WORK=${WORK:-/workspace/dc_rerun_20261026}
# ⚠️ 宿主/容器路径**必须分开给**（这条坑本方向已踩过 9 次以上）：日志在宿主落盘，容器里跑的是另一视角
export HOST_WORK=${HOST_WORK:-/data2/hliu553/dc_rerun_20261026}
export MODEL=${MODEL:-/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3}
export VIS_VAR=${VIS_VAR:-CUDA_VISIBLE_DEVICES}
export DC_BACKEND=${DC_BACKEND:-kunlun}
export DC_DIST_BT=${DC_DIST_BT:-cpu:gloo,cuda:flagcx}
# 单卡段固定用卡 0（10-26 同口径）；训练两卡用 0,7（10-27 11:40 普查：卡7 11.5GiB/0% util，
# 卡2/3 已 78GiB+100% 利用率、卡1 为已知故障卡 ⇒ 只有 0/7 适合与他人共跑，此处如实记录）
export DEVS=${DEVS:-0,7}
export MAX_STEPS=${MAX_STEPS:-50}

# ⚠️ 本实例的代码副本布局与仓内不同（宿主 `/data2/hliu553/dc_rerun_20261026/prototype` 平铺，
#    没有 `dev/device-context/` 这一层）⇒ 判据核心脚本路径**显式给**，不靠相对路径推。
export CORE=${CORE:-/data2/hliu553/dc_rerun_20261026/prototype/scripts/run_legparity_dcmodel.sh}
[ -f "$CORE" ] || { echo "找不到判据核心脚本：$CORE（用 CORE=<路径> 显式给）" >&2; exit 2; }
exec bash "$CORE"
