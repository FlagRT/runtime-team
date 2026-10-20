#!/bin/bash
# 复现尝试：复刻 pre 阶段的前置状态（先跑 std_steps 的前 5 个探针），再跑审计
# 依据：pre 的 4 次 FAIL 都出现在"刚跑完这批探针"之后；15:0x 的对照实验没有这层前置
set -uo pipefail
NAME=hliu553-dc-dev
CINTP=/workspace/runtime-team/dev/device-context/prototype
HOSTSH=/bmcp_lvm_fs/hliu553/probes_ppu
CINTSH=/workspace/probes_ppu
OUT=$HOSTSH/l1_repro_20261010
mkdir -p "$OUT"

run_once () {
  local tag=$1
  docker exec $NAME bash -lc "
set -u
cd $CINTP || exit 9
export DC_BACKEND=ppu DC_ROOT=$CINTP CUDA_VISIBLE_DEVICES=0 DC_OUT_DIR=$CINTSH/l1_repro_20261010
echo '--- 前置探针（std_steps 前 5 步）---'
python3 -u probes/probe_stream_semantics_full.py > $CINTSH/l1_repro_20261010/pre_semantics.log 2>&1; echo \"  semantics rc=\$?\"
python3 -u probes/probe_graph_capture_stream_v2.py > $CINTSH/l1_repro_20261010/pre_graph.log 2>&1; echo \"  graph rc=\$?\"
python3 -u probes/probe_stream_quota.py > $CINTSH/l1_repro_20261010/pre_quota.log 2>&1; echo \"  quota rc=\$?\"
python3 -u probes/probe_bc_contract.py --backend ppu --out $CINTSH/l1_repro_20261010/pre_bc.json > $CINTSH/l1_repro_20261010/pre_bc.log 2>&1; echo \"  bc rc=\$?\"
echo '--- 立刻跑审计 x3 ---'
for i in 1 2 3; do
  timeout 900 python3 -u scripts/duty_response_audit.py --backend ppu --out $CINTSH/l1_repro_20261010/${tag}_run\$i.json > $CINTSH/l1_repro_20261010/${tag}_run\$i.log 2>&1
  echo \"  ${tag} run\$i: \$(grep -E 'DUTY_RESPONSE_AUDIT ppu' $CINTSH/l1_repro_20261010/${tag}_run\$i.log | tail -1)\"
  grep -E '^  \[(OK|FAIL|SKIP)\] L1 ' $CINTSH/l1_repro_20261010/${tag}_run\$i.log | tail -1 | cut -c1-46
done
"
}

echo "===== 当前 base.py 状态 ====="
grep -nE 'lambda: native_stream|self\._id_holder\(native_stream\)' \
  /bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype/runtime/backends/base.py | tail -2

echo "===== 尝试复现（弱引用版应更能复现，若当前非弱引用则先切换）====="
run_once repro

echo "===== 再跑一轮（确认是否稳定）====="
run_once repro2
echo L1_REPRO_DONE
