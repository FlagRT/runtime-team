#!/bin/bash
# 910C 第 10 轮回归（2026-10-08 · D4：「(A) 真落地后的补齐轮」）
#
# 为什么必须有这一轮：`4f3ff1a` 的 910C 证据只覆盖到「接口层落地」；此后 `ea503cc`
# 又动了**共享层**（`create_stream` 增加优先级分支 + **新增 `release_stream` / `owns_stream`** +
# `check_stream_usable` 收紧「已释放」拦截），而 910C 因**网络不可达**（2026-09-30 14:00–14:45）
# 未复跑 ⇒ 台账 **D4**。本轮把 D4 收掉。
#
# 破坏面（本轮改动，逐项列出）：
#   runtime/backends/base.py            —— create_stream 六条门禁 + owns_stream /
#                                          release_stream / _register_owned_stream / _destroy_stream_raw
#   runtime/api/stream.py               —— Stream.release() + check_stream_usable「已释放」拦截
#   runtime/__init__.py                 —— 导出 release_stream / stream_priority_readback
#   runtime/backends/kunlun/backend.py  —— （本机不覆盖，属 P800 破坏面）
#   runtime/backends/cambricon/backend.py —— （本机不覆盖，属 MLU590 破坏面）
#   scripts/backend_offline_check.py    —— [8b] / [8b-②] 判据段
#
# `create_stream()` 的**全部消费方**必须至少被覆盖一次（本脚本内标 ✓）：
#   proto_train_leg.py:170 ✓ · proto_infer_leg.py:107 ✓ · proto_error_recovery_loop.py:110 ✓
#   probes/recover_multiproc_stress.py ✓ · conformance/runner.py ✓ · smoke_runtime.py ✓
#   duty_response_audit.py ✓ · demos/demo_unified.py ✓ · probes/probe_stream_release_and_control.py ✓
#   proto_infer_serve.py → **需外部 vLLM ⇒ 单独一轮**（run_910c_serve_r10.sh）
#
# 免跑（理由可复核，且**按破坏面当轮重判**）：
#   serve_standard.sh —— 不经过本层（`grep -cE "runtime|create_stream" scripts/serve_standard.sh`
#   仅命中注释里的镜像名）；服务化形态改由 run_910c_serve_r10.sh 里「proto_infer_serve.py」
#   覆盖本层消费方。
set -u
PROTO=/mnt/raid/hliu553/dc_r10_20261008/prototype
OUT=/mnt/raid/hliu553/dc_r10_20261008/out_r10
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
MODEL=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B
DRIVER=/mnt/raid/hliu553/dc_r10_20261008/r10_regress_910c_npu_20261008.log
export DC_BACKEND=ascend
export PYTHONDONTWRITEBYTECODE=1
export DC_OUT_DIR=$OUT
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗（复核本轮关键修复点）====="
echo "-- 共享层新接口（各期望 1）"
grep -c "def _create_stream_raw" runtime/backends/base.py
grep -c "def owns_stream"        runtime/backends/base.py
grep -c "def release_stream"     runtime/backends/base.py
grep -c "def _register_owned_stream" runtime/backends/base.py
grep -c "def stream_priority_readback" runtime/__init__.py
echo "-- 已释放拦截（api/stream.py，期望 >=1）"
grep -c "已释放" runtime/api/stream.py
echo "-- 离线判据段（[8b] / [8b-②]）"
grep -c "8b" scripts/backend_offline_check.py
echo "-- 新增探针在不在"
ls -1 probes/probe_stream_release_and_control.py probes/selfcheck_root_resolution.py probes/recover_multiproc_stress.py
echo "-- create_stream 消费方盘点（消费方清单，防漏覆盖）"
grep -rln "create_stream(" runtime/ scripts/ probes/ --include='*.py' | sort

step () {
  local name="$1"; shift
  echo "===== $name ====="
  date "+[%H:%M:%S] start"
  "$@" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
  date "+[%H:%M:%S] end"
}

step r10_offline_ascend       $PY scripts/backend_offline_check.py --backend ascend
step r10_symmetry_all         $PY scripts/backend_offline_check.py --all
step r10_smoke_ascend         $PY runtime/smoke_runtime.py --backend ascend
step r10_conf13_ascend        $PY runtime/conformance/runner.py --backend ascend --out "$OUT/r10_conf13_ascend.json"
step r10_confinfer6_ascend    $PY runtime/conformance/runner.py --backend ascend --cases infer_cases --out "$OUT/r10_confinfer6_ascend.json"
step r10_coninvariants_ascend $PY runtime/conformance/runner.py --backend ascend --cases contract_invariants --out "$OUT/r10_coninvariants_ascend.json"
step r10_duty_ascend          $PY scripts/duty_response_audit.py --backend ascend --out "$OUT/r10_duty_ascend.json"
step r10_errorloop_ascend     $PY runtime/proto/proto_error_recovery_loop.py --backend ascend --out "$OUT/r10_errorloop_ascend.json"
step r10_bc_probe_ascend      $PY probes/probe_bc_contract.py --backend ascend --out "$OUT/r10_bc_probe_ascend.json"
step r10_prio_api_ascend      $PY probes/probe_stream_priority_api.py --backend ascend --dev 0 --out "$OUT/r10_prio_api_ascend.json"
step r10_entry_verify_ascend  $PY probes/recover_entry_verify.py --backend ascend --dev 0 --out "$OUT/r10_entry_verify_ascend.json"
step r10_prio_release_ascend  $PY probes/probe_stream_release_and_control.py --backend ascend --dev 0 --out "$OUT/r10_prio_release_ascend.json"
step r10_root_resolution      $PY probes/selfcheck_root_resolution.py
step r10_stream_semantics     $PY probes/probe_stream_semantics_full.py
step r10_demo_unified         $PY runtime/demos/demo_unified.py

# ⚠️ 配额探针带 `DC_QUOTA_N=2000`：真实上限 1979 已由 B2 单独立项，
#    这里只要"正常档位仍可用"，不要把配额探针跑成大额分配。
echo "===== r10_stream_quota(2000) ====="
date "+[%H:%M:%S] start"
env DC_QUOTA_N=2000 $PY probes/probe_stream_quota.py > "$OUT/r10_stream_quota.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"

# ── 多卡多进程故障恢复压测（A2 型；3 rank = 容器内 0/1/2）──
echo "===== r10_multiproc_stress ====="
date "+[%H:%M:%S] start"
$PY probes/recover_multiproc_stress.py --backend ascend --devices 0,1,2 --rounds 30 \
    --out "$OUT/a2_r10" > "$OUT/r10_multiproc_stress.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"

# ── 两条腿（create_stream 的真实消费方；2 卡 = 容器内 0,1）──
# ⚠️ **首跑后修正（2026-10-08）**：本段原先漏了 `DC_ROOT=$PROTO`，导致 `import runtime` 失败
#    （`proto_train_leg.py` 的 `DC_ROOT` 默认值指向**宿主旧克隆**的 `dev/device-context`，不是本次原型根）。
#    首跑的原始失败原文与处置见 `../../docs/ASCEND_910C_R10_RERUN_20261008.md` §3；此处已补为显式前置条件。
echo "===== r10_train_leg ====="
date "+[%H:%M:%S] start"
( cd "$PROTO/runtime/proto" && env DC_ROOT=$PROTO ASCEND_RT_VISIBLE_DEVICES=0,1 DC_MODEL=$MODEL \
    DC_OUT_DIR=$OUT/train_npu MAX_STEPS=50 BATCH=4 SEQ=128 \
    $PY -m torch.distributed.run --standalone --nproc_per_node=2 --master_addr=127.0.0.1 \
    proto_train_leg.py > "$OUT/r10_train_leg.log" 2>&1 )
echo "rc=$?"
date "+[%H:%M:%S] end"

echo "===== r10_infer_leg ====="
date "+[%H:%M:%S] start"
( cd "$PROTO" && env DC_MODEL=$MODEL DC_ROUNDS=5 \
    $PY runtime/proto/proto_infer_leg.py > "$OUT/r10_infer_leg.log" 2>&1 )
echo "rc=$?"
date "+[%H:%M:%S] end"

echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
