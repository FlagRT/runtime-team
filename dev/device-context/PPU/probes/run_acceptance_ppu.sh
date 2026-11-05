#!/bin/bash
# =============================================================================
# PPU 全量验收 · 单入口（第四实例）—— 在宿主 ai-server 上执行
#
# 为什么要这个脚本（2026-10-26）：
#   PPU 接入期的 13 步是分散在 5 个分项脚本 + 若干手工命令里跑的；而 910C / MLU590
#   都有单入口（`910C/probes/run_910c_p9_acceptance.sh` 等）。本脚本把 PPU 的
#   **全量调通重跑**收成一个入口，逐段留 rc 与 verdict，供「基线复跑零回归」类判据直接引用。
#
# 设计要点
#   · 分段可续跑：`--only s03,s04` 只跑指定段；`--list` 列出全部段名
#   · 逐段记录 rc：⚠️ **判据通过 ≠ 进程干净退出**（PPU 退出期段错误曾表现为 rc=139）⇒
#     每段同时记 rc，SIGSEGV/SEGV 单独标出
#   · 证据按 run-id 分目录（历史批次永久保留，不 rm -rf 旧证据）
#   · 路径纪律：HO 由容器视角 O 按挂载映射推出（/bmcp_lvm_fs/hliu553 ↔ /workspace），不手写
#   · 选卡按**当前实测**：单卡段默认 DEV=0（2026-10-26 10:57 实测 ppu0 已用 5.5GiB、
#     ppu1 已用 8.3GiB ⇒ 服务化 0.9 利用率需 88.5GiB，两张都够但 dev0 余量更大）
#
# 用法
#   bash run_acceptance_ppu.sh                 # 全部段
#   bash run_acceptance_ppu.sh --list          # 列段名
#   bash run_acceptance_ppu.sh --only s04,s05  # 只跑指定段
#   RUNID=myrun bash run_acceptance_ppu.sh     # 自定 run-id
# =============================================================================
set -u

C=hliu553-dc-dev
P=/workspace/runtime-team/dev/device-context/prototype
O=/workspace/probes_ppu/acceptance_ppu                # 容器视角
HO=/bmcp_lvm_fs/hliu553/probes_ppu/acceptance_ppu     # 宿主视角
PY=/usr/local/bin/python3
# ⚠️ 宿主侧脚本路径（委派 T3/T456 用）：它们在**宿主**上执行，不能用容器视角的 /workspace/…
HP=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/PPU/probes
DEV=${DEV:-0}          # 单卡段（实测选卡，见文件头）
DEVS=${DEVS:-0,1}      # 训练腿两卡
MODEL=${MODEL:-/workspace/models/Qwen3-Embedding-0.6B}

ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list) sed -n 's/^# @step \([a-z0-9]*\) \(.*\)$/  \1  \2/p' "$0"; exit 0 ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "未知参数 $1" >&2; exit 2 ;;
  esac
done

RUNID=${RUNID:-$(date +%Y%m%d_%H%M%S)}
O="$O/$RUNID"; HO="$HO/$RUNID"
mkdir -p "$HO"
SUMMARY="$HO/summary.tsv"
printf "step\trc\tverdict\n" > "$SUMMARY"
echo "run-id: $RUNID"
echo "证据目录（宿主侧）: $HO"
echo

want() { [ -z "$ONLY" ] && return 0; case ",$ONLY," in *",$1,"*) return 0;; esac; return 1; }

# 跑一步：$1=段名 $2=日志名 $3=容器内命令（bash -lc 传入）
step() {
  local id="$1" log="$2" cmd="$3"
  want "$id" || return 0
  echo "===== [$id] $(date '+%H:%M:%S') ====="
  # ⚠️ DC_MODEL 必须显式传：proto_infer_leg.py 的默认值是 910C 专有路径
  #    （/mnt/raid/hliu553/…），在 PPU/MLU590 上会报 HFValidationError（2026-10-26 实测）
  docker exec -e DC_BACKEND=ppu -e DC_ROOT="$P" -e DC_OUT_DIR="$O" -e DC_MODEL="$MODEL" \
      -e CUDA_VISIBLE_DEVICES="$DEV" -e PYTHONDONTWRITEBYTECODE=1 \
      "$C" bash -lc "cd $P && $cmd" > "$HO/$id-$log.log" 2>&1
  local rc=$?
  # verdict：从日志里抓一行结论词；抓不到就记 rc
  local v
  v=$(grep -aoE "(CONFORMANCE_PASS|CONTRACT_INVARIANTS_PASS|SMOKE_PASS|TRAIN_LEG_PASS|INFER_LEG_PASS|SERVE_STANDARD_PASS|SERVE_LEG_PASS|ERROR_RECOVERY_LOOP_PASS|DUTY_RESPONSE_PASS|STREAM_SEMANTICS_PASS|GRAPH_CAPTURE_PASS|STREAM_QUOTA_PASS|SCHED_EFFECT|PASS|FAIL)" "$HO/$id-$log.log" 2>/dev/null | tail -1)
  [ -z "$v" ] && v="(no-verdict)"
  # 退出码异常单独标出
  local mark=""
  case "$rc" in
    0) ;;
    139|134|11) mark="⚠️SIGSEGV" ;;
    *) mark="⚠️rc=$rc" ;;
  esac
  printf "%s\t%s%s\t%s\n" "$id" "$rc" "$mark" "$v" >> "$SUMMARY"
  echo "  rc=$rc$mark  verdict=$v  （日志 $id-$log.log）"
  grep -aE "汇总|结论|PASS|FAIL|通过|失败" "$HO/$id-$log.log" 2>/dev/null | tail -3 | sed 's/^/    /'
  echo
}

# ---------------------------------------------------------------- 无设备段
# @step s01 离线契约自检（无设备）
step s01 offline "DC_BACKEND=ppu $PY scripts/backend_offline_check.py --backend ppu"

# @step s02 跨后端对称性自检（四家，无设备）
step s02 symmetry "$PY scripts/backend_offline_check.py --all"

# ---------------------------------------------------------------- 单卡段
# @step s03 真机 smoke
step s03 smoke "$PY -u runtime/smoke_runtime.py --backend ppu"

# @step s04 conformance 13 例
step s04 conf13 "$PY -u runtime/conformance/runner.py --backend ppu --out $O/s04_conf13.json"

# @step s05 conformance 推理 6 例
step s05 confinfer6 "$PY -u runtime/conformance/runner.py --backend ppu --cases infer_cases --out $O/s05_confinfer6.json"

# @step s06 契约不变式 I1–I4
step s06 invariants "$PY -u runtime/conformance/runner.py --backend ppu --cases contract_invariants --out $O/s06_invariants.json"

# ---------------------------------------------------------------- 两条腿
# @step s07 训练腿 2 卡 50 步
if want s07; then
  echo "===== [s07] $(date '+%H:%M:%S') ====="
  docker exec -e DC_BACKEND=ppu -e DC_ROOT="$P" -e DC_OUT_DIR="$O" \
      -e CUDA_VISIBLE_DEVICES="$DEVS" -e PYTHONDONTWRITEBYTECODE=1 \
      -e MAX_STEPS=50 -e BATCH=4 -e SEQ=128 -e DC_MODEL="$MODEL" \
      "$C" bash -lc "cd $P && $PY -m torch.distributed.run --standalone --nproc_per_node=2 runtime/proto/proto_train_leg.py" \
      > "$HO/s07-trainleg.log" 2>&1
  rc=$?
  v=$(grep -aoE "TRAIN_LEG_PASS [0-9]+/[0-9]+|TRAIN_LEG_FAIL" "$HO/s07-trainleg.log" | tail -1)
  mark=""; case "$rc" in 0);; 139|134|11) mark="⚠️SIGSEGV";; *) mark="⚠️rc=$rc";; esac
  printf "s07\t%s%s\t%s\n" "$rc" "$mark" "${v:-(no-verdict)}" >> "$SUMMARY"
  echo "  rc=$rc$mark  verdict=${v:-(no-verdict)}"
  grep -aE "loss|tok/s|TRAIN_LEG" "$HO/s07-trainleg.log" | tail -4 | sed 's/^/    /'
  echo
fi

# @step s08 推理腿（单卡前向 13/13）
step s08 inferleg "$PY -u runtime/proto/proto_infer_leg.py --backend ppu --out $O/s08_infer_leg.json"

# ---------------------------------------------------------------- 服务化（委派既有分项脚本：form + 消费方 + 清理）
# @step s09 服务化 form + 消费方腿（委派 run_t3，含停机清理）
if want s09; then
  echo "===== [s09] $(date '+%H:%M:%S') ====="
  DEV=$DEV RUNID="$RUNID-t3" bash "$HP/run_t3_ppu_serve_20261014.sh" \
      > "$HO/s09-serve.log" 2>&1
  rc=$?
  # ⚠️ 结论在委派脚本的**嵌套证据目录**里（10-26 实测：只读外层日志会得到 no-verdict）
  T3D=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t3_20261014/$RUNID-t3
  v=$(grep -aohE "SERVE_STANDARD_PASS|SERVE_LEG_PASS \| [0-9]+/[0-9]+|SERVE_LEG_FAIL" \
        "$HO/s09-serve.log" "$T3D"/form/*.log "$T3D"/consumer/*.log 2>/dev/null | tr "\n" " " | tail -c 120)
  printf "s09\t%s\t%s\n" "$rc" "${v:-(no-verdict)}" >> "$SUMMARY"
  echo "  rc=$rc  verdict=${v:-(no-verdict)}"
  grep -aE "SERVE|ready|smoke|句/s|区分度" "$HO/s09-serve.log" | tail -5 | sed 's/^/    /'
  echo
fi

# ---------------------------------------------------------------- T4/T5/T6（委派既有分项脚本）
# @step s10 错误闭环 + 多流 8 项 + D2 调度效果（委派 run_t456）
if want s10; then
  echo "===== [s10] $(date '+%H:%M:%S') ====="
  DEV=$DEV RUNID="$RUNID-t456" bash "$HP/run_t456_ppu_20261014.sh" \
      > "$HO/s10-t456.log" 2>&1
  rc=$?
  T456D=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t456_20261014/$RUNID-t456
  v=$(grep -aohE "ERROR_RECOVERY_LOOP_PASS[^\n]*|STREAM_SEMANTICS_PASS[^\n]*|STREAM_PRIORITY_SCHED_EFFECT_PASS[^\n]*" \
        "$HO/s10-t456.log" "$T456D"/*.log 2>/dev/null | tr "\n" " " | tail -c 200)
  printf "s10\t%s\t%s\n" "$rc" "${v:-(no-verdict)}" >> "$SUMMARY"
  echo "  rc=$rc  verdict=${v:-(no-verdict)}"
  grep -aE "closed_loops|PASS|effect" "$HO/s10-t456.log" | tail -5 | sed 's/^/    /'
  echo
fi

# ---------------------------------------------------------------- 图捕获 / 配额 / 职责审计
# @step s11 图捕获（4/4）
step s11 graphcapture "$PY -u probes/probe_graph_capture_stream_v2.py"

# @step s12 流配额（3/3）
step s12 quota "$PY -u probes/probe_stream_quota.py"

# @step s13 职责响应审计（78 项）
step s13 duty "$PY -u scripts/duty_response_audit.py --backend ppu --out $O/s13_duty.json"

# ---------------------------------------------------------------- 汇总
echo "=========== 汇总（$SUMMARY）==========="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
n_fail=$(awk -F'\t' 'NR>1 && $2!=0' "$SUMMARY" | wc -l)
n_all=$(( $(wc -l < "$SUMMARY") - 1 ))
if [ "$n_fail" -eq 0 ]; then echo "overall: PASS（$n_all 段全部 rc=0）"; exit 0
else echo "overall: FAIL（$n_fail/$n_all 段 rc≠0）"; cat "$SUMMARY" | awk -F'\t' 'NR>1 && $2!=0' | sed 's/^/  /'; exit 1; fi
