#!/bin/bash
# MLU590 · 补齐轮 m1 第 3 组：两条腿（2026-10-08）
# **在容器内执行**（dc-mlu590-hliu553）。
#
# 覆盖依据 = `MLU590/docs/MLU590_FIX_WORKPACK_20261008.md` §3 第 3 组（序 17/18；
#   序 19「多进程 real 压测」如实跳过 —— 本实例【已实测确认不具备】`recovery_real`）。
#
# 三处必须显式给出的前提（缺一即出错或"看起来通过"）：
#   ① `DC_ROOT=$PROTO` —— proto_train_leg.py 的默认根指向**宿主旧克隆路径**（该机不存在）
#      ⇒ 不给就 `ModuleNotFoundError: No module named 'runtime'`；
#   ② `DC_DIST_BT=cncl` —— 寒武纪的集合通信后端名**必须实测、不可从另两家类推**；
#      脚本**刻意不给默认值**且不给就 exit 2（防 `gloo` 静默退化为纯 CPU 通信）。
#      ⇒ 本条腿自带的 `comm_all_reduce / all_gather / p2p`（**在设备张量上**）
#        就是"后端名真的生效"的判据，不需要额外探针；
#   ③ `DC_MODEL` 必须指到 **snapshots/<hash>/**（给缓存根会 `Unrecognized model`）。
set -u

PROTO=/work/dc_mlu_regen_20261008/prototype
OUT=/work/dc_mlu_regen_20261008/out_m1
PY=/flagos/bin/python3
MODEL=$(ls -d /hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/ | head -1)
# 用卡：当次（12:03:19）逐卡普查 —— 卡 0（他人 mooncake 96 MiB）、卡 1（他人 160 MiB）避开；
# 卡 2–7 全 0 MiB 无进程 ⇒ 训练腿取 2,3，推理腿取 2。
TRAIN_CARDS=2,3
INFER_CARD=2

export DC_BACKEND=cambricon
export DC_ROOT=$PROTO
export DC_MODEL=$MODEL
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/train_mlu"

echo "===== [0] 前提复核 ====="
echo "DC_MODEL = $DC_MODEL"
echo "训练腿用卡 = $TRAIN_CARDS ；推理腿用卡 = $INFER_CARD"
echo "-- 步 0：模型完整性（sha256 应与 HF blob 名一致；本轮已在开工前验过，此处复核）"
sha256sum "$MODEL/model.safetensors"
echo "-- 设备可见性"
$PY -c "
import torch, torch_mlu
print('mlu calc  =', float(torch.ones(3, device='mlu:0').sum().item()))
print('dc_dist_bt =', __import__('os').environ.get('DC_DIST_BT', '<unset>'))
"

echo
echo "===== m1_train_leg（2 卡 · 50 步）====="
date "+[%H:%M:%S] start"
( cd "$PROTO/runtime/proto" && env MLU_VISIBLE_DEVICES=$TRAIN_CARDS DC_DIST_BT=cncl \
    MAX_STEPS=50 BATCH=4 SEQ=128 DC_OUT_DIR=$OUT/train_mlu \
    $PY -m torch.distributed.run --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
    proto_train_leg.py > "$OUT/m1_train_leg.log" 2>&1 )
echo "train rc=$?"
date "+[%H:%M:%S] end"
grep -aE "TRAIN_LEG|loss=|comm_" "$OUT/m1_train_leg.log" | tail -6

echo
echo "===== m1_infer_leg（单卡）====="
date "+[%H:%M:%S] start"
( cd "$PROTO" && env MLU_VISIBLE_DEVICES=$INFER_CARD DC_ROUNDS=5 \
    $PY runtime/proto/proto_infer_leg.py > "$OUT/m1_infer_leg.log" 2>&1 )
echo "infer rc=$?"
date "+[%H:%M:%S] end"
grep -aE "INFER_LEG|向量:|句/s" "$OUT/m1_infer_leg.log" | tail -6

echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
