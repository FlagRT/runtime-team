#!/usr/bin/env bash
set -u
C=dc-biren-hliu553
S=/usr/local/birensupa/sdk/1.10.0.0-24145.lkg
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
LDP=$(docker exec "$C" bash -c "find $S -name '*.so*' 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -")
echo "ld dirs: $(echo "$LDP" | tr ':' '\n' | wc -l)"
docker exec -i -e "LD_LIBRARY_PATH=$LDP" -e PYTHONDONTWRITEBYTECODE=1 "$C" python3 - <<'PY' 2>&1 | grep -viE '^\s*warn|UserWarning|Could not create log|COULD NOT' | head -18
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
print("supa has: device_name=%s mem_get_info=%s props=%s" % (
 hasattr(torch.supa,"get_device_name"), hasattr(torch.supa,"mem_get_info"), hasattr(torch.supa,"get_device_properties")))
PY
