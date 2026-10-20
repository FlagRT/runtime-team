#!/bin/bash
# PPU · T4（错误闭环）+ T5（多流语义 8 项）+ T6（D2 调度效果）—— 在宿主（ai-server）上执行
# ⚠️ 三项都要独占设备：必须等 T3 的服务完全停掉后再跑（T3 的 [C] 段负责清理）
# 路径纪律：HO 由 O 按挂载映射推出（/bmcp_lvm_fs/hliu553 ↔ /workspace），不得手写
set -u

C=hliu553-dc-dev
P=/workspace/runtime-team/dev/device-context/prototype
O=/workspace/probes_ppu/dc_t456_20261014
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t456_20261014
DEV=${DEV:-1}
PY=/usr/local/bin/python3

# ⚠️ **2026-10-14 修（自审发现）**：原实现 `rm -rf "$HO"` 会**删掉整个证据目录** ⇒
#    复跑一次就把上一批原始证据全部销毁（同族：另有 5 个驱动脚本用 `: >`/`rm -rf` 覆盖落盘）。
#    现改为**按 run-id 分目录**：新批次落在 $HO/<runid>/，历史批次永久保留。
RUNID=${RUNID:-$(date +%Y%m%d_%H%M%S)}
HO=$HO/$RUNID
mkdir -p "$HO"
echo "证据目录（本轮）: $HO"
echo "===== 前置：确认无服务残留并显示用卡 ====="
docker exec "$C" bash -lc 'ps -eo pid,cmd | grep -E "[v]llm|[s]erve_standard" | wc -l'
docker exec "$C" bash -lc "CUDA_VISIBLE_DEVICES=$DEV $PY -c \"import torch;f,t=torch.cuda.mem_get_info(0);print('dev free=%.1f GiB / total %.1f'%(f/2**30,t/2**30))\""

echo
echo "===== T4 错误闭环四类注入 ====="
date "+[%H:%M:%S] start"
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e DC_OUT_DIR=$O -e CUDA_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $P && $PY -u runtime/proto/proto_error_recovery_loop.py --backend ppu --out $O/t4_errorloop_ppu.json" \
    > "$HO/t4_errorloop_ppu.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"
grep -aE "closed_loops|skipped|failed|verdict|如实跳过|不适" "$HO/t4_errorloop_ppu.log" 2>/dev/null | tail -8
$PY - <<PYEOF 2>/dev/null || true
import json
try:
    d = json.load(open("$HO/t4_errorloop_ppu.json"))
    print("  JSON: closed_loops=%s skipped=%s failed=%s verdict=%s" % (d.get("closed_loops"), d.get("skipped"), d.get("failed"), d.get("verdict")))
    for r in d.get("results", []):
        print("   ", r.get("inject"), "|", r.get("category"), "|", r.get("disposition"), "| ok=", r.get("ok"), "| skipped=", r.get("skipped"))
except Exception as e:
    print("  JSON 读取失败:", e)
PYEOF

echo
echo "===== T5 多流语义探针（8 项）====="
date "+[%H:%M:%S] start"
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e DC_OUT_DIR=$O -e DC_TAG=_ppu -e CUDA_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $P && $PY -u probes/probe_stream_semantics_full.py --rounds 5" \
    > "$HO/t5_stream_semantics_ppu.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"
grep -aE "STREAM_SEMANTICS|PASS|FAIL" "$HO/t5_stream_semantics_ppu.log" 2>/dev/null | tail -8
cp -a "$HO"/stream_semantics_full_result*.json "$HO/t5_stream_semantics_full_result.json" 2>/dev/null || true

echo
echo "===== T6 D2 流优先级「调度效果」（含正对照 + 分辨力判据）====="
date "+[%H:%M:%S] start"
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e DC_OUT_DIR=$O -e CUDA_VISIBLE_DEVICES=$DEV -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $P && $PY -u probes/probe_stream_priority_sched_effect.py --dev 0 --out $O/t6_sched_effect_ppu.json" \
    > "$HO/t6_sched_effect_ppu.log" 2>&1
echo "rc=$?"
date "+[%H:%M:%S] end"
grep -aE "场景|净效应|结论|胜率|EFFECT|分辨力|正对照" "$HO/t6_sched_effect_ppu.log" 2>/dev/null | tail -14

echo
echo "===== 产物清单 ====="
ls -la "$HO" | tail -12
echo "T456_DONE"
