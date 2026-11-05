#!/bin/bash
# 910C 框架格修复验证（第二段）：post7 overlay 下我方**推理腿**读数是否变化
set -u
C=dc-probe-910c-1104
PY=/usr/local/python3.11.15/bin/python3
P=/workspace/dev/device-context/prototype
O=/workspace/fw_ops_out
docker exec "$C" bash -lc "mkdir -p $O"
echo "=== 臂 B: 推理腿 × torch_npu 2.10.0.post7（环境层 overlay，镜像未改）==="
docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 -e DC_BACKEND=ascend -e DC_ROOT=$P \
  -e DC_MODEL=/workspace/models/Qwen3-Embedding-0.6B -e DC_OUT_DIR=$O \
  -e PYTHONPATH=/workspace/pkgs_npu_post7 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  "$C" bash -lc "cd $P && $PY -u runtime/proto/proto_infer_leg.py --backend ascend" 2>&1 | grep -aE "INFER_LEG_(PASS|FAIL)|区分度|句/s|torch_npu|Error|error" | tail -8
