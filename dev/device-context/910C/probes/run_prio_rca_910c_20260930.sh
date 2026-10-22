#!/bin/bash
# 串行跑两件事：① torch 侧配额行为；② 按官方机制重测优先级
set -u
C=dc-quota-probe-0930
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
P=/mnt/raid/hliu553/dc_legs_20260929/prototype
B=/mnt/raid/hliu553/dc_legs_20260929
echo "===== [1] torch 侧配额行为（CAP=4000）====="
MODE=torch CAP=4000 $PY -u $B/find_stream_quota_910c.py 2>&1 | grep -avE "^\[W|Warning"
echo "===== [2] 按官方机制重测优先级（gate 模式）====="
DC_BACKEND=ascend DC_OUT_DIR=$B/rca_out $PY -u $P/probes/probe_stream_priority_queued.py --dev 0 2>&1 | grep -avE "^\[W|Warning"
echo "===== ALL DONE ====="
