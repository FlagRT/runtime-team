#!/bin/bash
# =============================================================================
# 平头哥 PPU（第四家实例）· `DC_MODEL` 两腿口径复验（2026-10-27）
#   —— 共享层改动：训练腿补 resolve_model
#
# 缺陷：`proto_infer_leg` 有 `resolve_model()`（HF 缓存根 → snapshots/<hash>），
#       `proto_train_leg` 没有 ⇒ 同一份 `DC_MODEL=/hf_cache/hub/models--Qwen--…`
#       推理腿过、训练腿报 `ValueError: Unrecognized model in …`（10-27 MLU590 实测暴露）。
# 修法：抽成 `runtime/proto/model_path.py` 两腿共用 + `scripts/model_path_parity_check.py` 静态同源守卫。
#
# 本文件只是**薄封装**：判据本体在 `prototype/scripts/run_legparity_dcmodel.sh`（四实例共用一份）。
#
# 环境事实（沿用 `run_acceptance_ppu.sh` 的实测约定）：
#   容器 hliu553-dc-dev · 容器内 /workspace == 宿主 /bmcp_lvm_fs/hliu553
#   解释器 /usr/local/bin/python3
#   选卡 CUDA_VISIBLE_DEVICES（容器内 0 基；PPU 走 torch.cuda 命名空间）
#   ⚠️ 训练腿必须 `DC_DIST_BT=nccl`（**不是 pccl**：库名 ≠ 后端名），且容器带
#      `-e NCCL_SOCKET_IFNAME=bond0`（镜像写死 eth0 而本机无 eth0）
#   模型 /workspace/models/Qwen3-Embedding-0.6B（本实例为普通目录形态 ⇒ 合成缓存由核心脚本造）
#
# 用法（**宿主 ai-server 上跑**，脚本自己 docker exec）：
#   bash run_ppu_legparity_20261027.sh
# =============================================================================
set -u
export C=${C:-hliu553-dc-dev}
export PY=${PY:-/usr/local/bin/python3}
export PROTO=${PROTO:-/workspace/runtime-team/dev/device-context/prototype}
export WORK=${WORK:-/workspace/probes_ppu/legparity}
# ⚠️ 宿主/容器路径**必须分开给**（这条坑本方向已踩过 9 次以上）：日志在宿主落盘，容器里跑的是另一视角
export HOST_WORK=${HOST_WORK:-/bmcp_lvm_fs/hliu553/probes_ppu/legparity}
export MODEL=${MODEL:-/workspace/models/Qwen3-Embedding-0.6B}
export VIS_VAR=${VIS_VAR:-CUDA_VISIBLE_DEVICES}
export DC_BACKEND=${DC_BACKEND:-ppu}
export DC_DIST_BT=${DC_DIST_BT:-nccl}
export DEVS=${DEVS:-0,1}
export MAX_STEPS=${MAX_STEPS:-50}

CORE=${CORE:-$(cd "$(dirname "$0")/../../prototype/scripts" 2>/dev/null && pwd)/run_legparity_dcmodel.sh}
[ -f "$CORE" ] || { echo "找不到判据核心脚本：$CORE（用 CORE=<路径> 显式给）" >&2; exit 2; }
exec bash "$CORE"
