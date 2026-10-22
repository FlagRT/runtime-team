#!/bin/bash
# 910C 第 9 轮回归（2026-09-30 · (A) 方案：流优先级统一 API 落地轮）
#
# 破坏面（本轮改动）：
#   runtime/backends/base.py          —— create_stream 模板方法（**所有流创建都走新路径**）
#                                        + _priority_bounds + stream_priority_readback 默认
#   runtime/backends/{ascend,kunlun,cambricon}/backend.py —— 能力键 + _create_stream_raw
#   runtime/__init__.py               —— create_stream(priority=) + stream_priority_readback 导出
#   runtime/conformance/contract_invariants.py —— 入口映射（I1④）
#   scripts/backend_offline_check.py  —— stub 真形 + [8b] 判据
#   runtime/smoke_runtime.py          —— MockBackend 跟随抽象方法改名
#
# `create_stream()` 的**全部消费方**必须至少被覆盖一次（改动落在它们的路径上）：
#   proto_train_leg.py:170 ✓本脚本 · proto_infer_leg.py:107 ✓ · proto_error_recovery_loop.py:110 ✓
#   conformance/runner.py ✓ · smoke_runtime.py ✓ · duty_response_audit.py ✓ · demos/demo_unified.py ✓
#   proto_infer_serve.py:89 → **需外部 vLLM 服务 ⇒ 单独一轮**（见 run_910c_serve_leg_20260930.sh）
#
# 免跑（理由可复核）：serve_standard.sh **不经过本层**
#   （`grep -E "runtime|create_stream" scripts/serve_standard.sh` = 0 真实命中，仅 1 处注释里的镜像名）
set -u
PROTO=/mnt/raid/hliu553/dc_legs_20260929/prototype
OUT=/mnt/raid/hliu553/dc_legs_20260929/out_r9
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
MODEL=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B
export DC_BACKEND=ascend
export PYTHONDONTWRITEBYTECODE=1
export DC_OUT_DIR=$OUT
mkdir -p "$OUT"
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗 ====="
grep -c "def _create_stream_raw" runtime/backends/base.py
grep -c "stream_priority_control" runtime/backends/base.py
grep -c "def stream_priority_readback" runtime/__init__.py
grep -c "\[8b\]" scripts/backend_offline_check.py
grep -c "def _create_stream_raw" runtime/proto/proto_train_leg.py 2>/dev/null || true  # 期望 0

step () {
  local name="$1"; shift
  echo "===== $name ====="
  "$@" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
}

step r9_offline_ascend        $PY scripts/backend_offline_check.py --backend ascend
step r9_symmetry_all          $PY scripts/backend_offline_check.py --all
step r9_smoke_ascend          $PY runtime/smoke_runtime.py --backend ascend
step r9_conf13_ascend         $PY runtime/conformance/runner.py --backend ascend --out "$OUT/r9_conf13_ascend.json"
step r9_confinfer6_ascend     $PY runtime/conformance/runner.py --backend ascend --cases infer_cases --out "$OUT/r9_confinfer6_ascend.json"
step r9_coninvariants_ascend  $PY runtime/conformance/runner.py --backend ascend --cases contract_invariants --out "$OUT/r9_coninvariants_ascend.json"
step r9_duty_ascend           $PY scripts/duty_response_audit.py --backend ascend --out "$OUT/r9_duty_ascend.json"
step r9_errorloop_ascend      $PY runtime/proto/proto_error_recovery_loop.py --backend ascend --out "$OUT/r9_errorloop_ascend.json"
step r9_bc_probe_ascend       $PY probes/probe_bc_contract.py --backend ascend --out "$OUT/r9_bc_probe_ascend.json"
step r9_prio_api_ascend       $PY probes/probe_stream_priority_api.py --backend ascend --dev 0 --out "$OUT/r9_prio_api_ascend.json"
step r9_stream_semantics      $PY probes/probe_stream_semantics_full.py
step r9_stream_quota          env DC_QUOTA_N=2000 $PY probes/probe_stream_quota.py
step r9_demo_unified          $PY runtime/demos/demo_unified.py

# ── 两条腿（create_stream 的真实消费方；2 卡 = davinci2,3）──
echo "===== r9_train_leg ====="
( cd "$PROTO/runtime/proto" && env ASCEND_RT_VISIBLE_DEVICES=0,1 DC_MODEL=$MODEL \
    DC_OUT_DIR=$OUT/train_npu MAX_STEPS=50 BATCH=4 SEQ=128 \
    $PY -m torch.distributed.run --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
    proto_train_leg.py > "$OUT/r9_train_leg.log" 2>&1 ); echo "rc=$?"

echo "===== r9_infer_leg ====="
( cd "$PROTO" && env DC_MODEL=$MODEL DC_ROUNDS=5 \
    $PY runtime/proto/proto_infer_leg.py > "$OUT/r9_infer_leg.log" 2>&1 ); echo "rc=$?"

echo "===== ALL DONE ====="
