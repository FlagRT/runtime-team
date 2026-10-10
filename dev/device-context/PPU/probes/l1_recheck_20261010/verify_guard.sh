#!/bin/bash
set -uo pipefail
NAME=hliu553-dc-dev
HOSTP=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype
CINTP=/workspace/runtime-team/dev/device-context/prototype
HOSTSH=/bmcp_lvm_fs/hliu553/probes_ppu
CINTSH=/workspace/probes_ppu
OUT=$HOSTSH/l1_guard_20261010
mkdir -p "$OUT"

echo "=== 同步最新 prototype（含类型复核）==="
rsync -a --exclude '__pycache__' --exclude '.DS_Store' "$HOSTP/../prototype/" 2>/dev/null || true
grep -n "tname" "$HOSTP/runtime/backends/base.py" | head -3

for mode in weak strong; do
  echo "===== 模式 = $mode ====="
  /usr/bin/python3 "$HOSTSH/switch_holder.py" "$mode"
  grep -nE "lambda: native_stream|self\._id_holder\(native_stream\), type" "$HOSTP/runtime/backends/base.py" | tail -1
  for i in 1 2 3; do
    docker exec $NAME bash -lc "cd $CINTP && CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu DC_ROOT=$CINTP timeout 1800 python3 -u $CINTSH/probe_l1_truth.py" > "$OUT/${mode}_$i.log" 2>&1
    l1=$(grep -vE "NCCL INFO" "$OUT/${mode}_$i.log" | grep -E "^\[(OK|FAIL)\] L1" | tail -1 | cut -c1-46)
    hit=$(grep -vE "NCCL INFO" "$OUT/${mode}_$i.log" | grep -E "^\[hits\]" | tail -1 | cut -c1-130)
    echo "  $mode#$i: $l1"
    echo "        $hit"
  done
done
/usr/bin/python3 "$HOSTSH/switch_holder.py" strong
echo GUARD_DONE
