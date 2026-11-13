#!/usr/bin/env bash
set -u
C=dc-biren-hliu553
S=/usr/local/birensupa/sdk/1.10.0.0-24145.lkg
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
LDP=$(docker exec "$C" bash -c "find $S -name '*.so*' 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -")
docker exec -i -e "LD_LIBRARY_PATH=$LDP" -e PYTHONDONTWRITEBYTECODE=1 "$C" python3 - <<'PY' 2>&1 | grep -viE '^\s*warn|UserWarning' | head -20
import torch, torch_br
print("torch:", torch.__version__, "| torch_br:", torch_br.__version__)
print("supa device_count:", torch.supa.device_count())
torch.supa.set_device(0)
x = torch.randn(64, 64).to("supa:0")
y = torch.randn(64, 64).to("supa:0")
z = x @ y
torch.supa.synchronize()
c = z.cpu()
print("matmul64 sum=%.4f nan=%d inf=%d" % (float(c.sum()), int(torch.isnan(c).sum()), int(torch.isinf(c).sum())))
import torch._C as Cm
for op in ["aten::mm", "aten::add.Tensor"]:
    t = Cm._dispatch_dump_table(op)
    print(op, "PU1:", [l.strip()[:70] for l in t.splitlines() if "PrivateUse1" in l][:1])
for fn in ["get_device_name", "mem_get_info", "get_device_properties"]:
    print(fn, ":", hasattr(torch.supa, fn))
PY
