#!/bin/bash
# L1 复核第 2 轮：修正探针路径 + 强/弱持有各重复 N 次，看 FAIL 是否偶发
set -uo pipefail
NAME=hliu553-dc-dev
HOSTP=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype
CINTP=/workspace/runtime-team/dev/device-context/prototype
HOSTSH=/bmcp_lvm_fs/hliu553/probes_ppu
CINTSH=/workspace/probes_ppu
OUT=$HOSTSH/l1_truth_20261010
N=${1:-20}
mkdir -p "$OUT"

run_probe () {
  local tag=$1
  docker exec $NAME bash -lc "cd $CINTP && CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu DC_ROOT=$CINTP timeout 1800 python3 -u $CINTSH/probe_l1_truth.py" > "$OUT/probe2_$tag.log" 2>&1
  echo "probe2_$tag rc=$?"
  grep -vE "NCCL INFO" "$OUT/probe2_$tag.log" | grep -E "^\[" | tail -14
}

run_audit_n () {
  local tag=$1 i line l1
  local fail=0
  for i in $(seq 1 $N); do
    docker exec $NAME bash -lc "cd $CINTP && CUDA_VISIBLE_DEVICES=0 DC_BACKEND=ppu DC_ROOT=$CINTP timeout 1200 python3 -u scripts/duty_response_audit.py --backend ppu --out $OUT/rep2_${tag}_run$i.json" > "$OUT/rep2_${tag}_run$i.log" 2>&1
    line=$(grep -E 'DUTY_RESPONSE_AUDIT ppu' "$OUT/rep2_${tag}_run$i.log" | tail -1)
    l1=$(grep -E '^  \[(OK|FAIL|SKIP)\] L1 ' "$OUT/rep2_${tag}_run$i.log" | tail -1 | cut -c1-4)
    echo "  $tag run$i: L1=$l1 | $line"
    case "$line" in *"FAIL 0"*) ;; *) fail=$((fail+1));; esac
  done
  echo "  ==> $tag：$N 次中 FAIL 出现 $fail 次"
}

echo "===== 当前锚点（应为强持有）====="
grep -n 'lambda: native_stream' "$HOSTP/runtime/backends/base.py"

echo "===== A. 强持有：probe + 审计 x$N ====="
run_probe strong
run_audit_n strong

echo "===== B. 切弱引用 ====="
/usr/bin/python3 "$HOSTSH/switch_holder.py" weak
grep -n 'self\._id_holder(native_stream)' "$HOSTP/runtime/backends/base.py" | tail -2

echo "===== C. 弱引用：probe + 审计 x$N ====="
run_probe weak
run_audit_n weak

echo "===== D. 恢复强持有 ====="
/usr/bin/python3 "$HOSTSH/switch_holder.py" strong
grep -n 'lambda: native_stream' "$HOSTP/runtime/backends/base.py"
echo L1_REPEAT_DONE
