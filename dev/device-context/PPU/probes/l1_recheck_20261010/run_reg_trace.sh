#!/bin/bash
# 抓「谁把默认流登记进 _owned_streams」：弱引用版下跑极轻量登记探针
set -uo pipefail
NAME=hliu553-dc-dev
HOSTP=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype
CINTP=/workspace/runtime-team/dev/device-context/prototype
HOSTSH=/bmcp_lvm_fs/hliu553/probes_ppu
CINTSH=/workspace/probes_ppu
OUT=$HOSTSH/l1_truth_20261010

echo "=== 切弱引用 ==="
/usr/bin/python3 "$HOSTSH/switch_holder.py" weak
grep -n 'self\._id_holder(native_stream)' "$HOSTP/runtime/backends/base.py" | tail -1

echo "=== 跑登记来源探针 ==="
docker exec $NAME bash -lc "cd $CINTP && CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu DC_ROOT=$CINTP timeout 1800 python3 -u $CINTSH/probe_reg_trace.py" > "$OUT/trace_weak.log" 2>&1
echo "rc=$?"
grep -vE "NCCL INFO" "$OUT/trace_weak.log" | tail -28

echo "=== 恢复强持有 ==="
/usr/bin/python3 "$HOSTSH/switch_holder.py" strong
grep -n 'lambda: native_stream' "$HOSTP/runtime/backends/base.py"
echo TRACE_DONE
