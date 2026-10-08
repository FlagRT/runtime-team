#!/bin/bash
# 910C 第 10 轮 · 两条腿补跑（2026-10-08）
# **在容器内执行**。
#
# 为什么要用替代模型路径：机上 `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B/model.safetensors`
# 已被**存储层静默损坏**（2 个张量含 34396 个非有限值，md127 RAID5 降级 [4/3]），
# 导致两条腿 `loss=nan` / `区分度 nan`，与设备上下文层无关（CPU 前向同样 NaN）。
# 修复文件落在**我方 scratch 目录**（未改动共享资产），并由 sha256 == HF blob 名
# `0437e45c…` 自证为**位级原件**。
set -u
PROTO=/mnt/raid/hliu553/dc_r10_20261008/prototype
OUT=/mnt/raid/hliu553/dc_r10_20261008/out_r10
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
MODEL=/mnt/raid/hliu553/dc_r10_20261008/models/Qwen3-Embedding-0.6B
export DC_BACKEND=ascend
export DC_MODEL=$MODEL
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1

echo "===== 替代模型完整性复核 ====="
sha256sum "$MODEL/model.safetensors"

echo "===== r10_train_leg（DC_ROOT 显式给出）====="
( cd "$PROTO/runtime/proto" && env DC_ROOT=$PROTO ASCEND_RT_VISIBLE_DEVICES=0,1 \
    DC_OUT_DIR=$OUT/train_npu MAX_STEPS=50 BATCH=4 SEQ=128 \
    $PY -m torch.distributed.run --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
    proto_train_leg.py > "$OUT/r10_train_leg.log" 2>&1 )
echo "train rc=$?"
grep -aE "TRAIN_LEG|loss" "$OUT/r10_train_leg.log" | tail -3

echo "===== r10_infer_leg ====="
( cd "$PROTO" && env DC_ROUNDS=5 $PY runtime/proto/proto_infer_leg.py > "$OUT/r10_infer_leg.log" 2>&1 )
echo "infer rc=$?"
grep -aE "INFER_LEG|向量:" "$OUT/r10_infer_leg.log" | tail -3

echo "===== ALL DONE ====="
