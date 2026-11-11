#!/usr/bin/env bash
set -u
C=dc-biren-hliu553
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
cat > /tmp/l2probe.py <<'PY'
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
print("mm PU1:", [l.strip()[:78] for l in t.splitlines() if "PrivateUse1" in l][:1])
print("supa has: device_name=%s mem_get_info=%s props=%s current_device=%s" % (
 hasattr(torch.supa,"get_device_name"), hasattr(torch.supa,"mem_get_info"),
 hasattr(torch.supa,"get_device_properties"), hasattr(torch.supa,"current_device")))
PY
docker cp /tmp/l2probe.py "$C":/tmp/l2probe.py >/dev/null 2>&1
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 /tmp/l2probe.py' 2>&1 | grep -viE '^\s*warn|UserWarning|Could not create log|COULD NOT' | head -16
docker exec "$C" rm -f /tmp/l2probe.py
rm -f /tmp/l2probe.py
