#!/usr/bin/env bash
set -u
C=dc-biren-hliu553
IMG='birensupa-smartinfer-vllm:26.07.09-py310-pt2.8.0-vllm0.21.0-c066s001t001b24145-br1xx'
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- census before create ---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo
docker rm -f "$C" >/dev/null 2>&1
mkdir -p "$HOME/hliu553_biren"
# 关键：不传 --device（default-runtime=biren 自动注入全部卡+库）；用 BIREN_VISIBLE_DEVICES 选物理卡 5,7
docker run -d --name "$C" \
  --network host \
  --shm-size 32g \
  -v "$HOME/hliu553_biren:/work" \
  -v /share/hf_cache:/share/hf_cache:ro \
  -e BIREN_VISIBLE_DEVICES=5,7 \
  --restart unless-stopped \
  "$IMG" /bin/bash -c 'sleep infinity' >/dev/null 2>&1
sleep 6
docker ps --filter "name=$C" --format '{{.Names}}|{{.Status}}'
echo '--- devices visible inside ---'
docker exec "$C" bash -c 'ls /dev/biren/ 2>/dev/null | tr "\n" " "; echo; echo VIS=$BIREN_VISIBLE_DEVICES'
echo '--- L2 minimal probe ---'
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 "$C" python3 - <<'PY' 2>&1 | grep -viE '^\s*warn|UserWarning|Could not create log|COULD NOT' | head -16
import torch, torch_br
print("torch:", torch.__version__, "| torch_br:", torch_br.__version__)
print("supa device_count:", torch.supa.device_count())
torch.supa.set_device(0)
x = torch.randn(64, 64).to("supa:0"); y = torch.randn(64, 64).to("supa:0")
z = x @ y
torch.supa.synchronize()
c = z.cpu()
print("matmul64 sum=%.4f nan=%d inf=%d" % (float(c.sum()), int(torch.isnan(c).sum()), int(torch.isinf(c).sum())))
import torch._C as Cm
t = Cm._dispatch_dump_table("aten::mm")
print("mm PU1:", [l.strip()[:75] for l in t.splitlines() if "PrivateUse1" in l][:1])
print("has: get_device_name=%s mem_get_info=%s get_device_properties=%s" % (
  hasattr(torch.supa,"get_device_name"), hasattr(torch.supa,"mem_get_info"), hasattr(torch.supa,"get_device_properties")))
PY
