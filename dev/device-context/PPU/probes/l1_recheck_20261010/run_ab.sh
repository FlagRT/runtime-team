#!/bin/bash
# 交替对照（A/B/A/B）：用「只 patch release」的探针确认复现器稳定性
# 依据：跨条件对照必须交替，否则无法区分变量效应与时间漂移
set -uo pipefail
NAME=hliu553-dc-dev
HOSTP=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype
CINTP=/workspace/runtime-team/dev/device-context/prototype
HOSTSH=/bmcp_lvm_fs/hliu553/probes_ppu
CINTSH=/workspace/probes_ppu
OUT=$HOSTSH/l1_truth_20261010
N=${1:-5}

for i in $(seq 1 $N); do
  for mode in weak strong; do
    /usr/bin/python3 "$HOSTSH/switch_holder.py" "$mode" > /dev/null
    docker exec $NAME bash -lc "cd $CINTP && CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu DC_ROOT=$CINTP timeout 1800 python3 -u $CINTSH/probe_l1_truth.py" > "$OUT/ab_${mode}_$i.log" 2>&1
    rc=$?
    l1=$(grep -vE "NCCL INFO" "$OUT/ab_${mode}_$i.log" | grep -E "^\[(OK|FAIL)\] L1" | tail -1 | cut -c1-40)
    hit=$(grep -vE "NCCL INFO" "$OUT/ab_${mode}_$i.log" | grep -E "^\[hits\]" | tail -1 | cut -c1-120)
    echo "$mode#$i rc=$rc | $l1"
    echo "      $hit"
  done
done
/usr/bin/python3 "$HOSTSH/switch_holder.py" strong > /dev/null
grep -n 'lambda: native_stream' "$HOSTP/runtime/backends/base.py"
echo AB_DONE
