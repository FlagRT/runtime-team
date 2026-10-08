#!/bin/bash
# MLU590（寒武纪 cambricon）· D2 轮（2026-10-08 第十三轮）：流优先级「调度效果」对照实验
# **在容器内执行**（dc-mlu590-hliu553）。
#
# 覆盖依据 = 台账 D2（`prototype/docs/OPEN_ITEMS_AUDIT_20260929.md`）：
#   「`priority` 的调度效果对照实验 —— 契约 §1.9 明确承诺的是接口能力、不是调度效果；
#     上层若据此做调度决策，需要性能侧证据」。
#
# ⚠️ 三处必须显式给出的前提（缺一即出错或"看起来通过"）：
#   ① `DC_ROOT=$PROTO` —— proto_*/demo 的默认根指向宿主旧克隆（该机不存在）；
#   ② `MLU_VISIBLE_DEVICES=2` —— 用卡当次普查（15:00:10）：卡 0（他人 mooncake 96 MiB）、
#      卡 1（他人 160 MiB）避开；卡 2–7 全 0 MiB 无进程。**调度效果实验必须独占卡**。
#   ③ 解释器 = 容器内 `/flagos/bin/python3`（Python 3.10.20 / torch 2.7.1 / torch_mlu 1.29.2）。
set -u

PROTO=/work/dc_mlu_regen_20261008/prototype
OUT=/work/dc_mlu_regen_20261008/out_d2
PY=/flagos/bin/python3
DRIVER=/work/dc_mlu_regen_20261008/d2_mlu590_20261008.log

export DC_BACKEND=cambricon
export DC_ROOT=$PROTO
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
export MLU_VISIBLE_DEVICES=2
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗 ====="
echo "-- 本轮新增的探针（期望存在且非空）"
ls -l probes/probe_stream_priority_sched_effect.py | awk '{print "   size="$5"  "$9}'
grep -c "GATE\|gate" probes/probe_stream_priority_sched_effect.py | sed 's/^/   gate 关键词命中数 = /'
echo "-- 共享层关键点（本轮**未改**，此处只是确认跑的是第十二轮修复后的版本）"
grep -c "def release_stream" runtime/backends/base.py
grep -c "holder() is native_stream" runtime/backends/base.py
grep -c "def stream_priority_readback" runtime/__init__.py
echo "-- 职责审计项数（期望 78）"
grep -c "@item(" scripts/duty_response_audit.py
echo "-- 设备可见性"
$PY -c "import torch,torch_mlu;print('mlu count =',torch.mlu.device_count(),' dev0 =',torch.mlu.get_device_name(0))"
echo "@ $(hostname) $(date '+%F %T')  可见设备 = $MLU_VISIBLE_DEVICES（容器内 0 基索引 = 物理卡 2）"

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
    --out "$OUT/d2_sched_effect_mlu590.json" > "$OUT/d2_sched_effect_mlu590.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"
grep -aE "STREAM_PRIORITY_SCHED_EFFECT|^  \[" "$OUT/d2_sched_effect_mlu590.log" | sed 's/^/   /'

echo
echo "########## 确认性回归（本轮**未改**共享层；按纪律复核一遍，防「顺带改坏」）##########"
step d2_offline_cambricon  $PY scripts/backend_offline_check.py --backend cambricon
step d2_symmetry_all       $PY scripts/backend_offline_check.py --all
step d2_smoke_cambricon    $PY runtime/smoke_runtime.py --backend cambricon
step d2_conf13_cambricon   $PY runtime/conformance/runner.py --backend cambricon --out "$OUT/d2_conf13_cambricon.json"
step d2_confinfer6_cambricon $PY runtime/conformance/runner.py --backend cambricon --cases infer_cases --out "$OUT/d2_confinfer6_cambricon.json"
step d2_coninvariants      $PY runtime/conformance/runner.py --backend cambricon --cases contract_invariants --out "$OUT/d2_coninvariants.json"
step d2_duty_cambricon     $PY scripts/duty_response_audit.py --backend cambricon --out "$OUT/d2_duty_cambricon.json"
step d2_selfcheck_duty_ext $PY probes/selfcheck_duty_audit_ext.py --backend cambricon --out "$OUT/d2_selfcheck_duty_ext.json"
step d2_prio_api_cambricon $PY probes/probe_stream_priority_api.py --backend cambricon --dev 0 --out "$OUT/d2_prio_api_cambricon.json"
step d2_prio_release_cambricon $PY probes/probe_stream_release_and_control.py --backend cambricon --dev 0 --out "$OUT/d2_prio_release_cambricon.json"
step d2_entry_verify       $PY probes/recover_entry_verify.py --backend cambricon --dev 0 --out "$OUT/d2_entry_verify.json"
step d2_root_resolution    $PY probes/selfcheck_root_resolution.py --out "$OUT/d2_root_resolution.json"

echo
echo "===== 各步结论 ====="
for f in d2_offline_cambricon d2_symmetry_all d2_smoke_cambricon d2_conf13_cambricon \
         d2_confinfer6_cambricon d2_coninvariants d2_duty_cambricon d2_selfcheck_duty_ext \
         d2_prio_api_cambricon d2_prio_release_cambricon d2_entry_verify d2_root_resolution; do
  printf "%-28s " "$f"
  grep -aoE "离线自检结果: [0-9]+ 通过 / [0-9]+ 失败 / [0-9]+ 跳过|对称性自检结果: [0-9]+ 通过 / [0-9]+ 失败|结果: [0-9]+ 通过 / [0-9]+ 失败|CONFORMANCE_[A-Z]+|CONTRACT_INVARIANTS_[A-Z]+|OK [0-9]+ / FAIL [0-9]+ / SKIP [0-9]+|DUTY_RESPONSE_[A-Z]+|SELFCHECK_DUTY_EXT_[A-Z]+|STREAM_PRIORITY_API_[A-Z]+|STREAM_RELEASE_CONTROL_[A-Z]+|ENTRY_VERIFY_[A-Z]+|ROOT_RESOLUTION_[A-Z]+|共 [0-9]+ 项：抓到 [0-9]+ · 本机不适用 [0-9]+ · \*\*未抓到 [0-9]+\*\*" \
      "$OUT/$f.log" 2>/dev/null | tr "\n" " "
  echo
done

echo
echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
