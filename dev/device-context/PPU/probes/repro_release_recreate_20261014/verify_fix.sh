#!/bin/bash
# PPU 修法验证（2026-10-14）：跑复现器，期望「不再复现」
set -u
C=hliu553-dc-dev
P=/workspace/runtime-team/dev/device-context/prototype
echo "=== 修法自证（base.py 内应有两处新逻辑）==="
docker exec "$C" bash -lc "cd $P && grep -c '_released_streams\"\)\.pop' runtime/backends/base.py; grep -c 'type(native_stream).__name__) *\$' runtime/backends/base.py"

echo
echo "=== 复现器 1：probe_who_holds.py（8 轮，期望无 RuntimeError）==="
docker exec -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 -e DC_ROOT=$P \
    "$C" bash -lc "cd $P && timeout 600 python3 -u /workspace/probes_ppu/probe_who_holds.py" 2>&1 | grep -vE "NCCL INFO" | tail -22

echo
echo "=== 复现器 2：probe_release_recreate_loop.py 60 轮（期望 NOT_REPRODUCED）==="
docker exec -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 -e DC_ROOT=$P \
    "$C" bash -lc "cd $P && timeout 900 python3 -u /workspace/probes_ppu/probe_release_recreate_loop.py 60" 2>&1 | grep -vE "NCCL INFO" | tail -6

echo
echo "=== 复现器 3：2×2 判据实验（期望使用点全 OK）==="
docker exec -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 -e DC_ROOT=$P \
    "$C" bash -lc "cd $P && timeout 600 python3 -u /workspace/probes_ppu/probe_weakref_2x2.py" 2>&1 | grep -vE "NCCL INFO" | tail -12
echo VERIFY_DONE
