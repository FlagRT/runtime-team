#!/bin/bash
# MLU590（寒武纪 cambricon）· 补齐轮 m1（2026-10-08）
#
# 定位：**第三家实例的首次全口径覆盖**。前两家（910C / P800）已在 2026-10-08 走到
#       「职责审计 78 项 + 非空转验证」口径，本实例此前停在 2026-09-28（旧口径 39 项）。
#
# 覆盖依据 = `MLU590/docs/MLU590_FIX_WORKPACK_20261008.md` §3 第 1 组 + 第 2 组。
#   · 第 1 组：离线/短项（14 项，无多卡需求）
#   · 第 2 组：三项"取数决定声明"的探针（B/C 契约 · 流优先级 · 流所有权/释放）
#
# ⚠️ 本脚本**不含**：两条腿（第 3 组）与服务化（第 4 组）—— 那两组各由独立脚本负责，
#    因为训练腿要 `DC_ROOT` 且在 `runtime/proto/` 下起，服务化必须换 vLLM 应用镜像的容器。
# ⚠️ 多进程 `real` 压测（工作包 §3 序 19）**如实跳过**：本实例【已实测确认不具备】
#    `recovery_real`（torch.mlu 下 reset*/destroy*/reinit* 全是内存统计类，无设备级重置原语）
#    ⇒ 见 `runtime/backends/cambricon/backend.py` 的 `_capabilities` 注释与 `probe_*` 结论。
#
# 用法（宿主侧）：docker exec -d <容器> bash -lc "bash /work/dc_mlu_regen_20261008/probes/run_mlu590_m1_20261008.sh"
set -u

PROTO=/work/dc_mlu_regen_20261008/prototype
OUT=/work/dc_mlu_regen_20261008/out_m1
PY=/flagos/bin/python3
DRIVER=/work/dc_mlu_regen_20261008/m1_mlu590_20261008.log

export DC_BACKEND=cambricon
export PYTHONDONTWRITEBYTECODE=1
export DC_OUT_DIR=$OUT
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗（这次跑的是 78 项职责审计口径）====="
echo "-- 统一面出口（各期望 1）"
grep -c "^def context_set" runtime/__init__.py
grep -c "^def stream_priority_range" runtime/__init__.py
echo "-- 职责审计项数（期望 78）"
grep -c "@item(" scripts/duty_response_audit.py
echo "-- 新域的域名字面量（各期望 1）"
grep -c '"H 内存句柄与生命周期"' scripts/duty_response_audit.py
grep -c '"M 统一 API 面"' scripts/duty_response_audit.py
echo "-- 流所有权/释放与优先级回读（各期望 1）"
grep -c "def owns_stream" runtime/backends/base.py
grep -c "def release_stream" runtime/backends/base.py
grep -c "def stream_priority_readback" runtime/__init__.py
echo "-- 离线自检的 [8b] / [8b-②] 段（期望 >=2）"
grep -c "8b" scripts/backend_offline_check.py

step () {
  local name="$1"; shift
  echo "===== $name ====="
  date "+[%H:%M:%S] start"
  "$@" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
  date "+[%H:%M:%S] end"
}

# ── 第 1 组 · 离线/短项 ──────────────────────────────────────────────
step m1_offline_cambricon    $PY scripts/backend_offline_check.py --backend cambricon
step m1_symmetry_all         $PY scripts/backend_offline_check.py --all
step m1_smoke_cambricon      $PY runtime/smoke_runtime.py --backend cambricon
step m1_conf13_cambricon     $PY runtime/conformance/runner.py --backend cambricon --out "$OUT/m1_conf13_cambricon.json"
step m1_confinfer6_cambricon $PY runtime/conformance/runner.py --backend cambricon --cases infer_cases --out "$OUT/m1_confinfer6_cambricon.json"
step m1_coninvariants        $PY runtime/conformance/runner.py --backend cambricon --cases contract_invariants --out "$OUT/m1_coninvariants.json"
step m1_duty_cambricon       $PY scripts/duty_response_audit.py --backend cambricon --out "$OUT/m1_duty_cambricon.json"
step m1_selfcheck_duty_ext   $PY probes/selfcheck_duty_audit_ext.py --backend cambricon --out "$OUT/m1_selfcheck_duty_ext.json"
step m1_errorloop_cambricon  $PY runtime/proto/proto_error_recovery_loop.py --backend cambricon --out "$OUT/m1_errorloop_cambricon.json"
step m1_entry_verify         $PY probes/recover_entry_verify.py --backend cambricon --dev 0 --out "$OUT/m1_entry_verify.json"
step m1_root_resolution      $PY probes/selfcheck_root_resolution.py --out "$OUT/m1_root_resolution.json"
step m1_stream_semantics     $PY probes/probe_stream_semantics_full.py
step m1_demo_unified         $PY runtime/demos/demo_unified.py

# 配额探针（DC_QUOTA_N 必须 <= 设备上限；不要跑成大额分配）
echo "===== m1_stream_quota(2000) ====="
date "+[%H:%M:%S] start"
env DC_QUOTA_N=2000 $PY probes/probe_stream_quota.py > "$OUT/m1_stream_quota.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"

# ── 第 2 组 · ⭐ 三项"取数决定声明"的探针 ────────────────────────────
step m1_bc_probe_cambricon   $PY probes/probe_bc_contract.py --backend cambricon --out "$OUT/m1_bc_probe_cambricon.json"
step m1_prio_api_cambricon   $PY probes/probe_stream_priority_api.py --backend cambricon --dev 0 --out "$OUT/m1_prio_api_cambricon.json"
step m1_prio_release_cambricon $PY probes/probe_stream_release_and_control.py --backend cambricon --dev 0 --out "$OUT/m1_prio_release_cambricon.json"

echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
