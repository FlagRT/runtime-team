#!/usr/bin/env bash
# 壁仞 BR_0 · 建本方向常驻容器（只挂自己用的卡 6；host 网络；家目录挂载）
# 选卡依据：11-11 10:30 普查——卡 0/2 故障、卡 1/3/4/5 被他人进程占用、卡 6/7 空闲
set -eu
C=dc-biren-hliu553
IMG='birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx'
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo "--- pre-flight: card 6 free? ---"
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '7p'
docker ps -a --filter "name=$C" --format '{{.Names}}|{{.Status}}' | head -2
if docker ps -a --format '{{.Names}}' | grep -q "^${C}$"; then
  echo "container $C already exists — start it"; docker start "$C"
else
  docker run -d --name "$C" \
    --network host \
    --device /dev/biren/card_6 \
    --device /dev/biren-m \
    -v "$HOME/hliu553_biren:/work" \
    -e BIREN_VISIBLE_DEVICES=0 \
    --restart unless-stopped \
    "$IMG" bash -lc 'sleep infinity'
fi
sleep 2
docker ps --filter "name=$C" --format '{{.Names}}|{{.Status}}'
echo '--- L2 最小判据（容器内、挂卡 6）---'
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" python3 - <<'PY' 2>&1 | grep -viE 'warning|warn\(' | head -20
import torch, torch_br
print("torch:", torch.__version__, "| torch_br:", torch_br.__version__)
print("supa device_count:", torch.supa.device_count())
torch.supa.set_device(0)
x = torch.randn(64, 64).to("supa:0")
y = torch.randn(64, 64).to("supa:0")
z = x @ y
torch.supa.synchronize()
c = z.cpu()
print("matmul 64x64: sum=%.4f  nan=%d  inf=%d" % (float(c.sum()), int(torch.isnan(c).sum()), int(torch.isinf(c).sum())))
import torch._C as C_
t = C_._dispatch_dump_table("aten::mm")
print("mm PU1:", [l.strip()[:80] for l in t.splitlines() if "PrivateUse1" in l][:1])
PY
echo '--- exit code above = L2 verdict ---'
