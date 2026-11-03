#!/bin/bash
# =============================================================================
# MLU590（第三实例）· 2026-11-03 共享层复跑 · 单入口
#
# 为什么要新写一份（而不是复用 run_mlu590_rerun_20261027.sh）：
#   ① 那份把结果写到**扁平** `$WH/out/`，复跑会**覆盖 10-27 的归档证据**（同族坑：覆盖式落盘静默销毁证据）；
#   ② 它内含 10-27 当时的临时绕过（`MODEL_SNAP_EXPR` 手工解析 snapshot），现在共享层已修，不该再带；
#   ③ 本轮要覆盖的改动面变了：`model_path.py` 两腿共用 + `serve_standard.sh` 分芯片默认 + 三个驱动脚本的
#      runid 目录修复 ⇒ 需要按 run-id 分目录、且**宿主视角与容器视角都分目录**（PPU 上刚踩过假 FAIL）。
#
# 覆盖范围（与 PPU 单入口 13 段同口径，缺的两段已在别处覆盖，见下）：
#   s01 离线契约自检 · s02 跨后端对称性 · s03 真机 smoke · s04 conformance 13 · s05 conformance 推理 6
#   s06 契约不变式 · s11 图捕获 · s12 流配额 · s13 职责响应审计(78)
#   ⚠️ **未在本脚本内重跑**：训练腿 50 步 / 推理腿 —— 已由同日 `run_mlu590_legparity_20261027.sh` 覆盖
#      （训练 6/6 · loss 15.4498→11.1479 两种传法各一遍；推理 13/13 · 区分度 0.6391 各一遍）；
#      服务化 —— 已由同日 `gmu_ab2_20261103_out/` 覆盖（0.9/0.4 两臂 10/10 + 默认值 `SERVE_STANDARD_PASS`）。
#      这是**范围裁定**，不是「已证明无回归」。
#
# 用法（宿主 Mlu-1 上跑，脚本自己 docker exec）：
#   bash run_mlu590_acceptance_20261103.sh                 # 全部段
#   RUNID=myrun bash run_mlu590_acceptance_20261103.sh     # 自定 run-id
# =============================================================================
set -u
C=dc-mlu590-hliu553
PY=/flagos/bin/python3
DEV=${DEV:-3}                          # 单卡段选卡（容器内 0 基）；本轮服务化占用卡 2 ⇒ 这里用 3
RUNID=${RUNID:-$(date +%Y%m%d_%H%M%S)}
# ⚠️ 三个路径必须分开算：代码副本在**基目录**下，证据才在 runid 目录下
#   （第一版把 P 拼成了 $WC/dev/... 而 WC 已含 acceptance 目录 ⇒ cd 失败 ⇒ 逐段 rc=1 且日志全空）
WB=/srv/hliu553/dc_rerun_20261027                       # 宿主基目录
CB=/work/dc_rerun_20261027                              # 容器基目录
P=$CB/dev/device-context/prototype                      # 代码副本（不含 runid）
HO=$WB/acceptance_20261103/$RUNID                       # 宿主视角证据
O=$CB/acceptance_20261103/$RUNID                        # 容器视角证据（与 HO 同步分目录）
mkdir -p "$HO"
SUMMARY="$HO/summary.tsv"
printf "step\trc\tverdict\n" > "$SUMMARY"

step() {  # step <段名> <日志名> <容器内命令>
  local name=$1 log=$2 cmd=$3 rc v
  echo "===== [$name] $(date '+%H:%M:%S') ====="
  docker exec -e MLU_VISIBLE_DEVICES=$DEV -e DC_BACKEND=cambricon -e DC_ROOT="$P" \
    -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -lc "cd $P && $cmd" > "$HO/$log.log" 2>&1
  rc=$?
  v=$(grep -aoE "CONFORMANCE_PASS|CONFORMANCE_FAIL|CONTRACT_INVARIANTS_(PASS|FAIL)|GRAPH_CAPTURE_(PASS|FAIL)[^:]*|STREAM_QUOTA_(PASS|FAIL)[^:]*|DUTY_RESPONSE_(PASS|FAIL)|SMOKE_[A-Z_]*PASS|[0-9]+ 通过 / [0-9]+ 失败[^|]*" "$HO/$log.log" 2>/dev/null | tail -1)
  printf "%s\t%s\t%s\n" "$name" "$rc" "${v:-(no-verdict)}" >> "$SUMMARY"
  echo "  rc=$rc  verdict=${v:-(no-verdict)}"
  grep -aE "verdict|PASS|FAIL" "$HO/$log.log" 2>/dev/null | tail -3 | sed 's/^/    /'
  echo
}

echo "############################################################"
echo "# MLU590 共享层复跑（2026-11-03）  $(date '+%F %T')"
echo "# 容器=$C ｜ 解释器=$PY ｜ 选卡 MLU_VISIBLE_DEVICES=$DEV ｜ RUNID=$RUNID"
echo "# 证据（宿主）=$HO"
echo "# 证据（容器）=$O"
echo "############################################################"
docker ps --format '{{.Names}}|{{.Status}}' | grep "$C" || true
docker exec "$C" bash -lc "$PY -c \"import torch, torch_mlu; print('torch', torch.__version__, 'torch_mlu', torch_mlu.__version__, 'mlu_cnt', torch.mlu.device_count())\""

step s01 offline    "$PY scripts/backend_offline_check.py --backend cambricon"
step s02 symmetry   "$PY scripts/backend_offline_check.py --all"
step s03 smoke      "$PY -u runtime/smoke_runtime.py --backend cambricon"
step s04 conf13     "$PY -u runtime/conformance/runner.py --backend cambricon --out $O/s04_conf13.json"
step s05 confinfer6 "$PY -u runtime/conformance/runner.py --backend cambricon --cases infer_cases --out $O/s05_confinfer6.json"
step s06 invariants "$PY -u runtime/conformance/runner.py --backend cambricon --cases contract_invariants --out $O/s06_invariants.json"
step s11 graphcap   "$PY -u probes/probe_graph_capture_stream_v2.py"
step s12 quota      "$PY -u probes/probe_stream_quota.py"
step s13 duty       "$PY scripts/duty_response_audit.py --backend cambricon --out $O/s13_duty.json"

echo "=========== 汇总（$SUMMARY）==========="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
n_fail=$(awk -F'\t' 'NR>1 && $2!=0' "$SUMMARY" | wc -l)
n_all=$(( $(wc -l < "$SUMMARY") - 1 ))
if [ "$n_fail" -eq 0 ]; then echo "overall: PASS（$n_all 段全部 rc=0）"; exit 0
else echo "overall: FAIL（$n_fail/$n_all 段 rc≠0）"; awk -F'\t' 'NR>1 && $2!=0' "$SUMMARY" | sed 's/^/  /'; exit 1; fi
