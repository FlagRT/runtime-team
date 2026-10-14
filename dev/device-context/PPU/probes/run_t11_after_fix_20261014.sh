#!/bin/bash
# T11 修后真机验证：同步修法到容器 → 复跑三个载体（不带任何注入）
set -u
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_20261014
O=/workspace/probes_ppu/dc_t11_20261014
P=/workspace/runtime-team/dev/device-context/prototype
C=hliu553-dc-dev
PY=/usr/local/bin/python3

echo "===== [0] 版本自证（容器内 sha256 必须等于本机）====="
docker exec "$C" bash -lc "cd $P && sha256sum runtime/backends/base.py && grep -c '_ledger_put' runtime/backends/base.py && grep -c 'cid is not None' runtime/backends/base.py"

echo
echo "===== [1] 载体复跑（修后，无注入）×2 轮交替 ====="
# ⚠️ 2026-10-14 修（自审发现）：原为清空 ⇒ 复跑销毁上一批记录。改为只追加。
echo "===== run=${RUNID:-$(date +%Y%m%d_%H%M%S)}  开始 $(date '+%F %T') =====" >> "$HO/after_matrix.log"
for r in 1 2; do
    for V in ours_default_dropped vendor_weakref_many; do
        docker exec -e DC_ROOT=$P -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 \
            -e PYTHONDONTWRITEBYTECODE=1 -e LD_PRELOAD=$O/t11_segv_handler.so \
            "$C" bash -lc "cd $P && $PY -u $O/t11_exit_segv_matrix.py $V" \
            > "$HO/after_${V}_r${r}.log" 2>&1
        printf "  round=%s carrier=%-22s rc=%s\n" "$r" "$V" "$?" | tee -a "$HO/after_matrix.log"
    done
    docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e CUDA_VISIBLE_DEVICES=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e LD_PRELOAD=$O/t11_segv_handler.so \
        "$C" bash -lc "cd $P && $PY -u runtime/proto/proto_error_recovery_loop.py --backend ppu --out $O/after_t4_r${r}.json" \
        > "$HO/after_t4_r${r}.log" 2>&1
    printf "  round=%s carrier=%-22s rc=%s\n" "$r" "t4_errorloop" "$?" | tee -a "$HO/after_matrix.log"
    docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e CUDA_VISIBLE_DEVICES=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e LD_PRELOAD=$O/t11_segv_handler.so \
        -e DC_OUT_DIR=$O -e DC_TAG=_after \
        "$C" bash -lc "cd $P && $PY -u probes/probe_stream_priority_sched_effect.py --dev 0 --out $O/after_t6_r${r}.json" \
        > "$HO/after_t6_r${r}.log" 2>&1
    printf "  round=%s carrier=%-22s rc=%s\n" "$r" "t6_sched" "$?" | tee -a "$HO/after_matrix.log"
done

echo
echo "===== [2] 判据行（修后必须与修前同结论）====="
grep -ah "ERROR_RECOVERY_LOOP" "$HO"/after_t4_r*.log | head -2
grep -ah "STREAM_PRIORITY_SCHED_EFFECT" "$HO"/after_t6_r*.log | head -2
echo "--- 是否仍有崩溃采集（应为空）---"
grep -al "T11_CRASH_CAPTURED" "$HO"/after_*.log 2>/dev/null || echo "  （无）"
echo
echo "===== [3] 前置缺口副作用：僵尸进程仍为 1（T3 停机残留，与本轮无关）====="
docker exec "$C" bash -lc 'ps -eo pid,stat,cmd | grep -E "[v]llm" | head -3'
