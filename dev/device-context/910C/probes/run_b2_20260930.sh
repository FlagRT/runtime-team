#!/bin/bash
# B2（工作包 D）取数总脚本 —— 优先级效果 + 配额真实上限
set -u
C=dc-quota-probe-0930
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
P=/mnt/raid/hliu553/dc_legs_20260929/prototype
O=/mnt/raid/hliu553/dc_legs_20260929/b2_out
mkdir -p "$O"
IN="docker exec"
echo "===== [0] 版本与环境 ====="
docker exec "$C" bash -lc "grep -c 'submit_order_dominates' $P/probes/probe_stream_priority_effect.py; $PY -c 'import torch,torch_npu;print(torch.__version__, torch_npu.__version__)'"

echo "===== [1] 优先级效果 ====="
$IN -e DC_BACKEND=ascend -e DC_OUT_DIR=$O "$C" "$PY" "$P/probes/probe_stream_priority_effect.py"

echo "===== [2] 配额真实上限（逐档递增）====="
for N in 4000 8000 16000 32000 64000 128000 256000; do
  echo "--- N=$N ---"
  $IN -e DC_BACKEND=ascend -e DC_QUOTA_N=$N -e DC_OUT_DIR=$O -e DC_TAG=_n$N "$C" "$PY" "$P/probes/probe_stream_quota.py" 2>&1 | grep -aE "Q1|STREAM_QUOTA"
done

echo "===== [3] 创建是否真实（资源代价）====="
$IN -e DC_BACKEND=ascend -e DC_QUOTA_N=1000000 -e DC_TOUCH_SAMPLE=500 -e DC_OUT_DIR=$O -e DC_TAG=_1M "$C" "$PY" "$P/probes/probe_stream_quota_touch.py"

echo "===== ALL DONE ====="
