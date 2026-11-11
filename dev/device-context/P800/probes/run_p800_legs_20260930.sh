#!/bin/bash
# P800 两条腿 + 服务化（2026-09-30）—— 共享层改动后的覆盖
# 选卡：dev4（292MiB/0%，最空闲）；次卡 dev0（1970MiB/0%，无进程）。
# ⛔ dev1 = 已知故障卡（UUID b3509946）：禁用。
set -u
C=hliu553-device-context-p800
PROTO=/workspace/dc_regress_20260929/prototype
OUT=/data2/hliu553/dc_regress_20260929/out_r6
OUT_C=/workspace/dc_regress_20260929/out_r6
MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
ENV="source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && cd $PROTO"
mkdir -p "$OUT"

echo "===== [1] 推理腿前向（单卡 dev4）====="
docker exec -e CUDA_VISIBLE_DEVICES=4 -e DC_BACKEND=kunlun -e DC_MODEL="$MODEL" \
  -e DC_OUT_DIR="$OUT_C" -e DC_ROUNDS=5 "$C" bash -lc "$ENV && python3 runtime/proto/proto_infer_leg.py" \
  > "$OUT/legs_infer_kunlun.log" 2>&1
echo "rc=$?"

echo "===== [2] 训练腿（2 卡 dev4,0 · 50 步）====="
docker exec -e CUDA_VISIBLE_DEVICES=4,0 -e DC_BACKEND=kunlun -e DC_MODEL="$MODEL" \
  -e DC_OUT_DIR="$OUT_C/train_kunlun" -e MAX_STEPS=50 -e BATCH=4 -e SEQ=128 \
  "$C" bash -lc "$ENV && mkdir -p $OUT_C/train_kunlun && python3 -m torch.distributed.run --standalone --nproc_per_node=2 --master_addr=127.0.0.1 proto_train_leg.py" \
  > "$OUT/legs_train_kunlun.log" 2>&1
echo "rc=$?"

echo "===== [3] 服务化（embedding · dev4 · TP=1 · EAGER=1）====="
docker exec -e DC_BACKEND=kunlun -e DEV=4 -e MODEL="$MODEL" -e SERVE_FORM=embed \
  -e STOP_AFTER=1 -e DC_OUT_DIR="$OUT_C/serve_embed" -e SMOKE_TIMEOUT=240 -e GPU_MEM_UTIL=0.25 \
  "$C" bash -lc "$ENV && mkdir -p $OUT_C/serve_embed && bash scripts/serve_standard.sh" \
  > "$OUT/serve_embed_kunlun.stdout" 2>&1
echo "rc=$?"
echo "===== ALL DONE ====="
