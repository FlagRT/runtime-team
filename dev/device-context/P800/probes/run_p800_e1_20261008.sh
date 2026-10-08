#!/bin/bash
# P800 · 台账 E1 落地轮（2026-10-08）：职责审计扩口径 + 补两处统一面出口
# **在容器内执行**（`hliu553-device-context-p800`）。
#
# 破坏面同 910C：`runtime/__init__.py` 只增两个出口 + 两个 `__all__` 名字；
# 职责审计 39 → 78 项。本机覆盖 `kunlun` 这一支。
#
# 免跑两条腿（理由可复核）：`runtime/proto/proto_*.py` 对新增的两个入口
# （`context_set` / `stream_priority_range`）**零引用**，且两处都是纯转发。
set -u
PROTO=/workspace/dc_e1_20261008/prototype
OUT=/workspace/dc_e1_20261008/out_e1
DRIVER=/workspace/dc_e1_20261008/e1_p800_20261008.log
export DC_BACKEND=kunlun
export PYTHONDONTWRITEBYTECODE=1
export DC_OUT_DIR=$OUT
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1

source /root/miniconda/etc/profile.d/conda.sh
conda activate python310_torch29_cuda
cd "$PROTO" || exit 1
PY=$(command -v python)

echo "===== [0] 跑的是当前版本吗 ====="
echo "-- python: $PY"
echo "-- 新增的两个统一面出口（各期望 1）"
grep -c "^def context_set" runtime/__init__.py
grep -c "^def stream_priority_range" runtime/__init__.py
echo "-- 职责审计项数（期望 78）"
grep -c "@item(" scripts/duty_response_audit.py
echo "-- 免跑两条腿的理由（各期望 0）"
grep -cE "context_set|stream_priority_range" runtime/proto/proto_train_leg.py || true
grep -cE "context_set|stream_priority_range" runtime/proto/proto_infer_leg.py || true

step () {
  local name="$1"; shift
  echo "===== $name ====="
  date "+[%H:%M:%S] start"
  "$@" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
  date "+[%H:%M:%S] end"
}

step e1_offline_kunlun        $PY scripts/backend_offline_check.py --backend kunlun
step e1_symmetry_all          $PY scripts/backend_offline_check.py --all
step e1_smoke_kunlun          $PY runtime/smoke_runtime.py --backend kunlun
step e1_conf13_kunlun         $PY runtime/conformance/runner.py --backend kunlun --out "$OUT/e1_conf13_kunlun.json"
step e1_confinfer6_kunlun     $PY runtime/conformance/runner.py --backend kunlun --cases infer_cases --out "$OUT/e1_confinfer6_kunlun.json"
step e1_coninvariants         $PY runtime/conformance/runner.py --backend kunlun --cases contract_invariants --out "$OUT/e1_coninvariants.json"
step e1_duty_kunlun           $PY scripts/duty_response_audit.py --backend kunlun --out "$OUT/e1_duty_kunlun.json"
step e1_selfcheck_duty_ext    $PY probes/selfcheck_duty_audit_ext.py --backend kunlun --out "$OUT/e1_selfcheck_duty_ext.json"
step e1_root_resolution       $PY probes/selfcheck_root_resolution.py

echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
