#!/bin/bash
# PPU · T5 补齐三项（S-13 多设备绑定 / S-16 流配额 / S-7 图捕获）——宿主侧执行
set -u
C=hliu553-dc-dev
CP=/workspace/runtime-team/dev/device-context/prototype
O=/workspace/probes_ppu/dc_t5_20261014
HO=/bmcp_lvm_fs/hliu553/probes_ppu/dc_t5_20261014
mkdir -p "$HO"

echo "=== 前置：无残留 + 两卡可见性 ==="
docker exec "$C" bash -lc 'ps -eo pid,cmd | grep -cE "[v]llm|[p]robe_"'
docker exec -e CUDA_VISIBLE_DEVICES=0,1 "$C" bash -lc "python3 -c \"import torch;print('visible=',torch.cuda.device_count())\""

echo
echo "=== S-13：多设备流绑定（两卡可见）——重跑 8 项流语义探针 ==="
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$CP -e DC_OUT_DIR=$O -e DC_TAG=_ppu_2card \
    -e CUDA_VISIBLE_DEVICES=0,1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $CP && timeout 600 python3 -u probes/probe_stream_semantics_full.py --rounds 5" \
    > "$HO/t5_2card_stream_semantics.log" 2>&1
echo "rc=$?"
grep -aE "S13|S12|STREAM_SEMANTICS" "$HO/t5_2card_stream_semantics.log" | tail -5

echo
echo "=== S-16：流数量配额（2000）==="
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$CP -e DC_OUT_DIR=$O -e DC_QUOTA_N=2000 \
    -e CUDA_VISIBLE_DEVICES=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $CP && timeout 900 python3 -u probes/probe_stream_quota.py" \
    > "$HO/t5_stream_quota.log" 2>&1
echo "rc=$?"
grep -aE "PASS|FAIL|配额|2000" "$HO/t5_stream_quota.log" | tail -6

echo
echo "=== S-7：图捕获流语义（V2）==="
docker exec -e DC_BACKEND=ppu -e DC_ROOT=$CP -e DC_OUT_DIR=$O -e DC_TAG=_ppu \
    -e CUDA_VISIBLE_DEVICES=1 -e PYTHONDONTWRITEBYTECODE=1 \
    "$C" bash -lc "cd $CP && timeout 900 python3 -u probes/probe_graph_capture_stream_v2.py" \
    > "$HO/t5_graph_capture_v2.log" 2>&1
echo "rc=$?"
grep -aE "GRAPH_CAPTURE|G[0-9]|PASS|FAIL" "$HO/t5_graph_capture_v2.log" | tail -10

echo
echo "=== 产物 ==="
ls -la "$HO" | tail -8
echo T5_EXTRA_DONE
