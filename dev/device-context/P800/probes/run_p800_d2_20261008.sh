#!/bin/bash
# P800（昆仑芯 kunlun）· D2 轮（2026-10-08 第十三轮）：流优先级「调度效果」对照实验
# **在容器内执行**（hliu553-device-context-p800；宿主 /data2/hliu553 = 容器 /workspace）。
#
# 覆盖依据 = 台账 D2（`prototype/docs/OPEN_ITEMS_AUDIT_20260929.md`）。
# ⚠️ 预期：本实例**优先级空间退化为单点**（`cuCtxGetStreamPriorityRange` ⇒ least=0, greatest=0）
#    ⇒ D2 主项应如实出 `NOT_APPLICABLE`；但**仪器有效性那两组仍要跑**（用默认流，不需要优先级）
#    —— 否则「不适用」与「探针根本没工作」分不开。
#
# 三处必须显式给出的前提：
#   ① `DC_ROOT=$PROTO` —— proto_*/demo 的默认根指向宿主旧克隆（该机不存在）；
#   ② `CUDA_VISIBLE_DEVICES=0` —— 当次普查 8 卡空闲；`dev1` 是**故障卡**（0 MiB/0% 与好卡同貌，
#      靠 UUID 避开）；用卡前按纪律**当次重探**；
#   ③ 解释器 = conda `python310_torch29_cuda`（torch 2.9.0+cu129）。
set -u

PROTO=/workspace/dc_e1_20261008/prototype
OUT=/workspace/dc_e1_20261008/out_d2
DRIVER=/workspace/dc_e1_20261008/d2_p800_20261008.log

source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
PY=python

export DC_BACKEND=kunlun
export DC_ROOT=$PROTO
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗 ====="
echo "-- 本轮新增的探针（期望存在且非空）"
ls -l probes/probe_stream_priority_sched_effect.py | awk '{print "   size="$5"  "$9}'
echo "-- 共享层关键点（本轮**未改**，确认跑的是第十二轮修复后的版本）"
grep -c "holder() is native_stream" runtime/backends/base.py
grep -c "def stream_priority_readback" runtime/__init__.py
echo "-- 职责审计项数（期望 78）"
grep -c "@item(" scripts/duty_response_audit.py
echo "-- 设备可见性与故障卡规避"
$PY -c "import torch;print('visible cards =',torch.cuda.device_count(),' dev0 =',torch.cuda.get_device_name(0))"
echo "@ $(hostname) $(date '+%F %T')  可见设备 = $CUDA_VISIBLE_DEVICES"

step () {
  local name="$1"; shift
  echo "===== $name ====="
  date "+[%H:%M:%S] start"
  "$@" > "$OUT/$name.log" 2>&1
  echo "rc=$?"
  date "+[%H:%M:%S] end"
}

echo
echo "########## 主项：D2 流优先级「调度效果」对照实验 ##########"
date "+[%H:%M:%S] start"
$PY -u probes/probe_stream_priority_sched_effect.py --dev 0 \
    --out "$OUT/d2_sched_effect_p800.json" > "$OUT/d2_sched_effect_p800.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"
grep -aE "STREAM_PRIORITY_SCHED_EFFECT|^  \[" "$OUT/d2_sched_effect_p800.log" | sed 's/^/   /'

echo
echo "########## 确认性回归（本轮**未改**共享层；按纪律复核一遍）##########"
step d2_offline_kunlun      $PY scripts/backend_offline_check.py --backend kunlun
step d2_symmetry_all        $PY scripts/backend_offline_check.py --all
step d2_smoke_kunlun        $PY runtime/smoke_runtime.py --backend kunlun
step d2_conf13_kunlun       $PY runtime/conformance/runner.py --backend kunlun --out "$OUT/d2_conf13_kunlun.json"
step d2_confinfer6_kunlun   $PY runtime/conformance/runner.py --backend kunlun --cases infer_cases --out "$OUT/d2_confinfer6_kunlun.json"
step d2_coninvariants       $PY runtime/conformance/runner.py --backend kunlun --cases contract_invariants --out "$OUT/d2_coninvariants.json"
step d2_duty_kunlun         $PY scripts/duty_response_audit.py --backend kunlun --out "$OUT/d2_duty_kunlun.json"
step d2_selfcheck_duty_ext  $PY probes/selfcheck_duty_audit_ext.py --backend kunlun --out "$OUT/d2_selfcheck_duty_ext.json"
step d2_prio_api_kunlun     $PY probes/probe_stream_priority_api.py --backend kunlun --dev 0 --out "$OUT/d2_prio_api_kunlun.json"
step d2_prio_release_kunlun $PY probes/probe_stream_release_and_control.py --backend kunlun --dev 0 --out "$OUT/d2_prio_release_kunlun.json"
step d2_entry_verify        $PY probes/recover_entry_verify.py --backend kunlun --dev 0 --out "$OUT/d2_entry_verify.json"
step d2_root_resolution     $PY probes/selfcheck_root_resolution.py --out "$OUT/d2_root_resolution.json"

echo
echo "===== 各步结论 ====="
for f in d2_offline_kunlun d2_symmetry_all d2_smoke_kunlun d2_conf13_kunlun \
         d2_confinfer6_kunlun d2_coninvariants d2_duty_kunlun d2_selfcheck_duty_ext \
         d2_prio_api_kunlun d2_prio_release_kunlun d2_entry_verify d2_root_resolution; do
  printf "%-28s " "$f"
  grep -aoE "离线自检结果: [0-9]+ 通过 / [0-9]+ 失败 / [0-9]+ 跳过|对称性自检结果: [0-9]+ 通过 / [0-9]+ 失败|结果: [0-9]+ 通过 / [0-9]+ 失败|CONFORMANCE_[A-Z]+|CONTRACT_INVARIANTS_[A-Z]+|OK [0-9]+ / FAIL [0-9]+ / SKIP [0-9]+|DUTY_RESPONSE_[A-Z]+|SELFCHECK_DUTY_EXT_[A-Z]+|STREAM_PRIORITY_API_[A-Z]+|STREAM_RELEASE_CONTROL_[A-Z]+|ENTRY_VERIFY_[A-Z]+|ROOT_RESOLUTION_[A-Z]+|共 [0-9]+ 项：抓到 [0-9]+ · 本机不适用 [0-9]+ · \*\*未抓到 [0-9]+\*\*" \
      "$OUT/$f.log" 2>/dev/null | tr "\n" " "
  echo
done

echo
echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
