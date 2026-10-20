#!/bin/bash
# 910C（昇腾 ascend）· D2 轮（2026-10-08 第十三轮）：流优先级「调度效果」对照实验
# **在容器内执行**（dc-d2-910c-20261008；宿主与容器路径同为 /mnt/raid/hliu553）。
#
# 覆盖依据 = 台账 D2（`prototype/docs/OPEN_ITEMS_AUDIT_20260929.md`）：
#   「`priority` 的调度效果对照实验 —— 契约 §1.9 明确承诺的是接口能力、不是调度效果；
#     上层若据此做调度决策，需要性能侧证据」。
#
# ⚠️ 预期（**先写死，不许事后解释**）：本实例**不声明 `stream_priority_control`**
#    （实测 `stream_priority=True` / `control=False` / `readback=True`，区间可读 `(7,0)`）
#    ⇒ D2 主项应如实出 `STREAM_PRIORITY_SCHED_EFFECT_NOT_APPLICABLE`，
#    **但仪器有效性那三组仍必须跑**（E3 工作量标定 / D1 分辨力 / V1 gate 有效性 / G2 正对照，
#    它们用**默认流**，不需要优先级）—— 否则「本机不适用」与「探针根本没工作」分不开。
#
# 四处必须显式给出的前提（缺一即出错或"看起来通过"）：
#   ① `DC_ROOT=$PROTO` —— proto_*/demo 的默认根指向宿主旧克隆；
#   ② `ASCEND_RT_VISIBLE_DEVICES=0` —— ⚠️ **容器内 0 基索引**（本容器只挂 `/dev/davinci2`
#      ⇒ 容器内就是 0；填 2 会得到 `aclInit 107001`，**看着像硬件坏了**）；
#   ③ 解释器 = `/mnt/raid/hliu553/venvs/venv-infer-a/bin/python`（torch 2.11.0 / torch_npu 2.11.0）；
#   ④ **独占卡**：性能类实验对同卡负载敏感。当次普查（15:31:39）16 卡全无运行进程，
#      他人占 davinci1(evalx)/6(yxy)/8-15(temp-cp)，`flaggems` 挂全 16 **不占**；
#      取 **davinci2**，本容器只挂它一张。
set -u

PROTO=/mnt/raid/hliu553/dc_d2_20261008/prototype
OUT=/mnt/raid/hliu553/dc_d2_20261008/out_d2
PY=/mnt/raid/hliu553/venvs/venv-infer-a/bin/python
DRIVER=/mnt/raid/hliu553/dc_d2_20261008/d2_910c_20261008.log

export DC_BACKEND=ascend
export DC_ROOT=$PROTO
export DC_OUT_DIR=$OUT
export PYTHONDONTWRITEBYTECODE=1
export ASCEND_RT_VISIBLE_DEVICES=0
mkdir -p "$OUT"
exec > "$DRIVER" 2>&1
cd "$PROTO" || exit 1

echo "===== [0] 跑的是当前版本吗 ====="
echo "-- 本轮新增的探针（期望存在且非空）"
ls -l probes/probe_stream_priority_sched_effect.py | awk '{print "   size="$5"  "$9}'
echo "-- 探针判据数（judge/skip 调用，期望 31）"
grep -cE 'judge\("|skip\("' probes/probe_stream_priority_sched_effect.py | sed 's/^/   /'
echo "-- 共享层关键点（本轮**未改**，此处只确认跑的是第十二轮修复后的版本）"
grep -c "def release_stream" runtime/backends/base.py | sed 's/^/   release_stream      = /'
grep -c "holder() is native_stream" runtime/backends/base.py | sed 's/^/   身份复核（第33条）= /'
grep -c "def stream_priority_readback" runtime/__init__.py | sed 's/^/   stream_priority_readback = /'
grep -c "def context_set" runtime/__init__.py | sed 's/^/   context_set（第30条）= /'
echo "-- 职责审计项数（期望 78）"
grep -c "@item(" scripts/duty_response_audit.py | sed 's/^/   /'
echo "-- 能力声明（决定 D2 的 applicable）"
$PY -c "import runtime; bk=runtime.use('ascend'); print('   supports =', {k: bk.supports(k) for k in ('stream_priority','stream_priority_control','stream_priority_readback')}); print('   range =', runtime.stream_priority_range())"
echo "-- 设备可见性"
$PY -c "import torch, torch_npu; print('   npu count =', torch.npu.device_count(), ' dev0 =', torch.npu.get_device_name(0))"
echo "@ $(hostname) $(date '+%F %T')  可见设备 = $ASCEND_RT_VISIBLE_DEVICES（容器内 0 基索引 = 物理卡 2）"

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
    --out "$OUT/d2_sched_effect_910c.json" > "$OUT/d2_sched_effect_910c.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"
grep -aE "STREAM_PRIORITY_SCHED_EFFECT|^  \[|^\[E1\]|不适用原因" "$OUT/d2_sched_effect_910c.log" | sed 's/^/   /'

echo
echo "########## 确认性回归（本轮**未改**共享层；按纪律复核一遍，防「顺带改坏」）##########"
step d2_offline_ascend      $PY scripts/backend_offline_check.py --backend ascend
step d2_symmetry_all        $PY scripts/backend_offline_check.py --all
step d2_smoke_ascend        $PY runtime/smoke_runtime.py --backend ascend
step d2_conf13_ascend       $PY runtime/conformance/runner.py --backend ascend --out "$OUT/d2_conf13_ascend.json"
step d2_confinfer6_ascend   $PY runtime/conformance/runner.py --backend ascend --cases infer_cases --out "$OUT/d2_confinfer6_ascend.json"
step d2_coninvariants       $PY runtime/conformance/runner.py --backend ascend --cases contract_invariants --out "$OUT/d2_coninvariants.json"
step d2_duty_ascend         $PY scripts/duty_response_audit.py --backend ascend --out "$OUT/d2_duty_ascend.json"
step d2_selfcheck_duty_ext  $PY probes/selfcheck_duty_audit_ext.py --backend ascend --out "$OUT/d2_selfcheck_duty_ext.json"
step d2_prio_api_ascend     $PY probes/probe_stream_priority_api.py --backend ascend --dev 0 --out "$OUT/d2_prio_api_ascend.json"
step d2_prio_release_ascend $PY probes/probe_stream_release_and_control.py --backend ascend --dev 0 --out "$OUT/d2_prio_release_ascend.json"
step d2_entry_verify        $PY probes/recover_entry_verify.py --backend ascend --dev 0 --out "$OUT/d2_entry_verify.json"
step d2_root_resolution     $PY probes/selfcheck_root_resolution.py --out "$OUT/d2_root_resolution.json"

echo
echo "===== 各步结论 ====="
for f in d2_offline_ascend d2_symmetry_all d2_smoke_ascend d2_conf13_ascend \
         d2_confinfer6_ascend d2_coninvariants d2_duty_ascend d2_selfcheck_duty_ext \
         d2_prio_api_ascend d2_prio_release_ascend d2_entry_verify d2_root_resolution; do
  printf "%-28s " "$f"
  grep -aoE "离线自检结果: [0-9]+ 通过 / [0-9]+ 失败 / [0-9]+ 跳过|对称性自检结果: [0-9]+ 通过 / [0-9]+ 失败|结果: [0-9]+ 通过 / [0-9]+ 失败|CONFORMANCE_[A-Z]+|CONTRACT_INVARIANTS_[A-Z]+|OK [0-9]+ / FAIL [0-9]+ / SKIP [0-9]+|DUTY_RESPONSE_[A-Z]+|SELFCHECK_DUTY_EXT_[A-Z]+|STREAM_PRIORITY_API_[A-Z]+|STREAM_RELEASE_CONTROL_[A-Z]+|ENTRY_VERIFY_[A-Z]+|ROOT_RESOLUTION_[A-Z]+|共 [0-9]+ 项：抓到 [0-9]+ · 本机不适用 [0-9]+ · \*\*未抓到 [0-9]+\*\*" \
      "$OUT/$f.log" 2>/dev/null | tr "\n" " "
  echo
done

echo
echo "===== ALL DONE ====="
date "+[%H:%M:%S]"
