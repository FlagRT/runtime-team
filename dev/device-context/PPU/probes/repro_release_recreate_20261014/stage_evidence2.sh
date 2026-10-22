#!/bin/bash
# 固化「修前 / 修后」对照证据（2026-10-14）——在宿主（ai-server）上执行
# 做法：备份修好版 → 把「新建即失效旧记录」那行临时换成 pass ⇒ 跑复现器（=修前）
#       → 从备份恢复 ⇒ 再跑复现器（=修后）。同一进程路径、同一机器、同一时段。
set -u
C=hliu553-dc-dev
HP=/bmcp_lvm_fs/hliu553/runtime-team/dev/device-context/prototype
CP=/workspace/runtime-team/dev/device-context/prototype
OUT=/bmcp_lvm_fs/hliu553/probes_ppu/repro_release_recreate_20261014
mkdir -p "$OUT/before" "$OUT/after"

run_probes () {
  local tag="$1"
  docker exec -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 -e DC_ROOT=$CP \
      "$C" bash -lc "cd $CP && timeout 900 python3 -u /workspace/probes_ppu/probe_release_recreate_loop.py 60" \
      2>&1 | grep -vE "NCCL INFO" > "$OUT/$tag/repro_loop.log"
  docker exec -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 -e DC_ROOT=$CP \
      "$C" bash -lc "cd $CP && timeout 600 python3 -u /workspace/probes_ppu/probe_weakref_2x2.py" \
      2>&1 | grep -vE "NCCL INFO" > "$OUT/$tag/repro_weakref_2x2.log"
  docker exec -e DC_BACKEND=ppu -e CUDA_VISIBLE_DEVICES=1 -e DC_ROOT=$CP \
      "$C" bash -lc "cd $CP && timeout 600 python3 -u /workspace/probes_ppu/probe_who_holds.py" \
      2>&1 | grep -vE "NCCL INFO" > "$OUT/$tag/repro_who_holds.log"
  echo "  [$tag] loop 结果 : $(grep -a '复现结果' "$OUT/$tag/repro_loop.log" | tail -1)"
  echo "  [$tag] 2x2 使用失败: $(grep -a '使用失败' "$OUT/$tag/repro_weakref_2x2.log" | tr '\n' ' ')"
  echo "  [$tag] 8 轮使用点 : $(grep -ac '使用点：✅' "$OUT/$tag/repro_who_holds.log") 次可用 / $(grep -ac '使用点：❌' "$OUT/$tag/repro_who_holds.log") 次失败"
}

echo "=== 0) 备份修好版 ==="
cp "$HP/runtime/backends/base.py" "$OUT/base_fixed_backup.py" && echo "  已备份（$(wc -l < "$OUT/base_fixed_backup.py") 行）"

echo
echo "=== 1) 修前（临时禁用「新建即失效旧记录」）==="
sed -i 's|self\._reg("_released_streams")\.pop(id(native_stream), None)|pass  # TEMP-DISABLED (before)|' \
    "$HP/runtime/backends/base.py"
echo "  禁用标记命中 = $(grep -c 'TEMP-DISABLED (before)' "$HP/runtime/backends/base.py")"
run_probes before

echo
echo "=== 2) 修后（从备份恢复）==="
cp "$OUT/base_fixed_backup.py" "$HP/runtime/backends/base.py"
echo "  恢复后残留禁用标记 = $(grep -c 'TEMP-DISABLED (before)' "$HP/runtime/backends/base.py" || true)"
echo "  恢复后 pop 行命中 = $(grep -c '_released_streams")\.pop' "$HP/runtime/backends/base.py" || true)"
run_probes after
echo EVIDENCE_DONE
