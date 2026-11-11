#!/bin/bash
# T11 取证 3：编译并加载**原生** SIGSEGV 处理器，采真崩溃现场（T4 / T6 两个载体）
set -u
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t11_20261014
O=/workspace/probes_ppu/dc_t11_20261014
P=/workspace/runtime-team/dev/device-context/prototype
C=hliu553-dc-dev
PY=/usr/local/bin/python3

echo "===== [A] 编译原生处理器（容器内 gcc）====="
docker exec "$C" bash -lc "cd $O && gcc -shared -fPIC -O1 -g -o t11_segv_handler.so t11_segv_handler.c -ldl 2>&1 && ls -l t11_segv_handler.so"

echo
echo "===== [B] 非空转自检：故意 SIGSEGV，必须被采集 ====="
docker exec -e LD_PRELOAD=$O/t11_segv_handler.so "$C" bash -lc \
    "$PY -u -c 'import ctypes; ctypes.string_at(0)'" > "$HO/native_selftest.log" 2>&1
echo "  selftest rc=$?"
sed -n '1,10p' "$HO/native_selftest.log"

echo
echo "===== [C] T4 载体 ×2 ====="
for i in 1 2; do
    docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e CUDA_VISIBLE_DEVICES=1 \
        -e PYTHONDONTWRITEBYTECODE=1 -e LD_PRELOAD=$O/t11_segv_handler.so \
        "$C" bash -lc "cd $P && $PY -u runtime/proto/proto_error_recovery_loop.py --backend ppu --out $O/t4_native$i.json" \
        > "$HO/t4_native$i.log" 2>&1
    echo "  T4#$i rc=$?"
done

echo
echo "===== [D] T6 载体 ×1 ====="
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$P -e CUDA_VISIBLE_DEVICES=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e LD_PRELOAD=$O/t11_segv_handler.so \
    -e DC_OUT_DIR=$O -e DC_TAG=_nat \
    "$C" bash -lc "cd $P && $PY -u probes/probe_stream_priority_sched_effect.py --dev 0 --out $O/t6_native.json" \
    > "$HO/t6_native.log" 2>&1
echo "  T6 rc=$?"

echo
echo "===== [E] 采集结果 ====="
for f in t4_native1 t4_native2 t6_native; do
    echo "----- $f -----"
    awk '/T11_CRASH_CAPTURED/{flag=1} flag' "$HO/$f.log" | head -16
done
