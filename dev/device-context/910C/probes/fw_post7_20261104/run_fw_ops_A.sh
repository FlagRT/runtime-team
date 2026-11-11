#!/bin/bash
# 对照臂 A：同容器同代码，仅 torch_npu 用镜像自带的 2.10.0（无 overlay）
set -u
C=dc-probe-910c-1104
PY=/usr/local/python3.11.15/bin/python3
P=/workspace/dev/device-context/prototype
O=/workspace/fw_ops_out_A
docker exec "$C" bash -lc "mkdir -p $O"
echo "=== 臂 A: 推理腿 × torch_npu 2.10.0（镜像自带，基线）==="
docker exec -e ASCEND_RT_VISIBLE_DEVICES=2 -e DC_BACKEND=ascend -e DC_ROOT=$P \
  -e DC_MODEL=/workspace/models/Qwen3-Embedding-0.6B -e DC_OUT_DIR=$O \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  "$C" bash -lc "cd $P && $PY -u runtime/proto/proto_infer_leg.py --backend ascend" 2>&1 | grep -aE "INFER_LEG_(PASS|FAIL)|区分度" | tail -4
echo "--- 两臂区分度/吞吐并排 ---"
for d in fw_ops_out_A:基线-2.10.0 fw_ops_out:post7; do
  dir=${d%%:*}; tag=${d##*:}
  printf "%-16s %s\n" "$tag" "$(docker exec "$C" bash -lc "grep -aoE 'INFER_LEG_PASS [0-9]+/[0-9]+|区分度 [0-9.]+|[0-9.]+ 句/s' /workspace/$dir/proto_infer_leg_result.json 2>/dev/null | tr '\n' ' '" 2>/dev/null)"
done
