#!/bin/bash
# 910C · 台账 E1 落地轮（2026-10-08）：职责审计扩口径 + 补两处统一面出口
#
# 破坏面（本轮改动，**只增不改**）：
#   runtime/__init__.py                 —— 新增 `context_set()` / `stream_priority_range()` 两个
#                                          统一面出口 + 两个 `__all__` 名字（契约 §1.7 / §1.9 早已承诺）
#   scripts/duty_response_audit.py      —— 新增 H–M 六个域（39 → 78 项）
#   probes/selfcheck_duty_audit_ext.py  —— 新增（非空转验证脚手架，本脚本内单独跑）
#
# 免跑两条腿（理由可复核）：`grep -c "context_set\|stream_priority_range" runtime/proto/proto_*.py`
#   = 0 ⇒ 两条腿**不经过**本轮新增的两个入口；且两处都是**纯转发**（一行 `current().X(...)`），
#   不改任何既有语义。故按「按破坏面覆盖」只跑无设备/短项 + 职责审计。
set -u
PROTO=/mnt/raid/hliu553/dc_e1_20261008/prototype
OUT=/mnt/raid/hliu553/dc_e1_20261008/out_e1
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
DRIVER=/mnt/raid/hliu553/dc_e1_20261008/e1_910c_npu_20261008.log
export DC_BACKEND=ascend
export PYTHONDONTWRITEBYTECODE=1
export DC_OUT_DIR=$OUT
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗 ====="
echo "-- 新增的两个统一面出口（各期望 1）"
grep -c "^def context_set" runtime/__init__.py
grep -c "^def stream_priority_range" runtime/__init__.py
echo "-- 职责审计项数（期望 78）"
grep -c "@item(" scripts/duty_response_audit.py
echo "-- 新域的域名字面量（各期望 1）"
grep -c '"H 内存句柄与生命周期"' scripts/duty_response_audit.py
grep -c '"M 统一 API 面"' scripts/duty_response_audit.py
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

step e1_offline_ascend       $PY scripts/backend_offline_check.py --backend ascend
step e1_symmetry_all         $PY scripts/backend_offline_check.py --all
step e1_smoke_ascend         $PY runtime/smoke_runtime.py --backend ascend
step e1_conf13_ascend        $PY runtime/conformance/runner.py --backend ascend --out "$OUT/e1_conf13_ascend.json"
step e1_confinfer6_ascend    $PY runtime/conformance/runner.py --backend ascend --cases infer_cases --out "$OUT/e1_confinfer6_ascend.json"
step e1_coninvariants        $PY runtime/conformance/runner.py --backend ascend --cases contract_invariants --out "$OUT/e1_coninvariants.json"
step e1_duty_ascend          $PY scripts/duty_response_audit.py --backend ascend --out "$OUT/e1_duty_ascend.json"
step e1_selfcheck_duty_ext   $PY probes/selfcheck_duty_audit_ext.py --backend ascend --out "$OUT/e1_selfcheck_duty_ext.json"
step e1_root_resolution      $PY probes/selfcheck_root_resolution.py

echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
