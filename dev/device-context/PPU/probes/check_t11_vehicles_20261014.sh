#!/bin/bash
# PPU · T11 同期阳性对照：两个「已知 rc=139」的载体今天是否仍复现
# 用途：若最小变体不崩，必须先用已知载体证明「本窗口测得出该现象」，否则不得写成「未复现」
set -u

C=hliu553-dc-dev
P=/workspace/runtime-team/dev/device-context/prototype
O=/workspace/probes_ppu/dc_t11_20261014
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_20261014
DEV=${DEV:-1}
PY=${PY:-/usr/local/bin/python3}

echo "===== [A] T6 优先级调度效果探针（10-14 已知 rc=139）====="
date "+[%F %T] start"
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e DC_OUT_DIR=$O -e DC_TAG=_t11 \
    -e CUDA_VISIBLE_DEVICES=$DEV -e PYTHONFAULTHANDLER=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $P && $PY -u probes/probe_stream_priority_sched_effect.py --dev 0 --out $O/t6_vehicle.json" \
    > "$HO/vehicle_t6.log" 2>&1
echo "T6 rc=$?" | tee -a "$HO/vehicle_rc.log"
date "+[%F %T] end"
grep -aE "STREAM_PRIORITY_SCHED_EFFECT|Fatal Python error|Segmentation fault" "$HO/vehicle_t6.log" | tail -6

echo
echo "===== [B] T4 错误闭环四类注入（10-14 已知 rc=139）====="
date "+[%F %T] start"
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e DC_OUT_DIR=$O \
    -e CUDA_VISIBLE_DEVICES=$DEV -e PYTHONFAULTHANDLER=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $P && $PY -u runtime/proto/proto_error_recovery_loop.py --backend ppu --out $O/t4_vehicle.json" \
    > "$HO/vehicle_t4.log" 2>&1
echo "T4 rc=$?" | tee -a "$HO/vehicle_rc.log"
date "+[%F %T] end"
grep -aE "ERROR_RECOVERY_LOOP|Fatal Python error|Segmentation fault" "$HO/vehicle_t4.log" | tail -6

echo
echo "===== [C] 落盘 ====="
ls -l "$HO" | head -12
