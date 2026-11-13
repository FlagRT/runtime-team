#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe3 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
cat > /tmp/ns_probe.py <<'PY'
import sys, torch, torch_br
print("torch:", torch.__version__)
print("torch_br:", getattr(torch_br, "__version__", "?"), torch_br.__file__)
for ns in ["biren", "supa", "br", "cuda", "xpu", "npu", "mlu"]:
    if hasattr(torch, ns):
        mod = getattr(torch, ns)
        extra = ""
        if hasattr(mod, "device_count"):
            try: extra = "device_count=%s" % mod.device_count()
            except Exception as e: extra = "device_count_err:%s" % type(e).__name__
        print("torch.%-6s exists %s" % (ns, extra))
import torch._C as C
print("dispatch_has_compile", getattr(C, "_dispatch_has_kernel_for_dispatch_key", None) is not None)
PY
for IMG in 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'; do
  echo ""; echo "############ $IMG (login shell) ############"
  docker run --rm --entrypoint bash "$IMG" -lc 'python3 /tmp/ns_probe.py' 2>&1 | head -18
done
echo '--- img2 entrypoint script: what does it set? ---'
docker run --rm --entrypoint bash 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107' -lc 'grep -E "export|PATH|LD_LIBRARY" /usr/local/birensupa/br_container_tools/biren_entrypoint.sh | head -12; echo; ls /usr/local/birensupa/ | head -12' 2>&1 | head -25
rm -f /tmp/ns_probe.py
echo "=== END probe3 ==="
