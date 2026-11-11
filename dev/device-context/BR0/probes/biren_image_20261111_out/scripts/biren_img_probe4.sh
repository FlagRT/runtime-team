#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe4 2026-11-11 ==="
for IMG in 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'; do
  echo ""; echo "############ $IMG ############"
  docker run --rm --entrypoint bash "$IMG" -lc 'pip list 2>/dev/null | grep -iE "supa|biren|br-|torch-br|flag" | head -8; echo "--- entrypoint.d ---"; ls /usr/local/birensupa/br_container_tools/entrypoint.d/ 2>/dev/null; echo "--- ns probe ---"; python3 -c "
import torch, torch_br
print(\"torch:\", torch.__version__)
print(\"torch_br:\", getattr(torch_br, \"__version__\", \"?\"), torch_br.__file__)
for ns in [\"biren\",\"supa\",\"br\",\"cuda\"]:
    if hasattr(torch, ns):
        m = getattr(torch, ns)
        try: extra = \"device_count=%s\" % m.device_count()
        except Exception as e: extra = \"dc_err:%s\" % type(e).__name__
        print(\"torch.%s: %s %s\" % (ns, \"\", extra))
"' 2>&1 | head -30
done
echo "=== END probe4 ==="
