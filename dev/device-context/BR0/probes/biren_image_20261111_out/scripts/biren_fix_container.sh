#!/usr/bin/env bash
set -u
C=dc-biren-hliu553
IMG='birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx'
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- census (pick free healthy) ---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null
echo '--- recreate: keep image ENTRYPOINT (libs injection), override CMD to stay alive ---'
docker rm -f "$C" >/dev/null 2>&1
mkdir -p "$HOME/hliu553_biren"
# 传参给 entrypoint：它注入环境后 exec 该命令驻留（与 xbench 的 Path=/bin/bash -c sleep infinity 同构）
docker run -d --name "$C" \
  --network host \
  --device /dev/biren/card_5 \
  --device /dev/biren/card_7 \
  --device /dev/biren-m \
  -v "$HOME/hliu553_biren:/work" \
  -e BIREN_VISIBLE_DEVICES=0,1 \
  --restart unless-stopped \
  "$IMG" /bin/bash -c 'sleep infinity' >/dev/null 2>&1
sleep 5
docker ps --filter "name=$C" --format '{{.Names}}|{{.Status}}'
echo '--- env inside (should have LD/ldconfig from entrypoint) ---'
docker exec "$C" bash -lc 'echo LD=${LD_LIBRARY_PATH:0:120}; ldconfig -p 2>/dev/null | grep -c supti; echo DEV=$BIREN_VISIBLE_DEVICES'
echo '--- import test ---'
docker exec "$C" bash -lc 'python3 -c "import torch, torch_br; print(torch.__version__, torch_br.__version__); print(\"supa_devices:\", torch.supa.device_count())"' 2>&1 | tail -3
