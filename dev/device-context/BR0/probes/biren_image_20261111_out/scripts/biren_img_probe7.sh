#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe7 2026-11-11 ==="
declare -A LIBDIR=(
  ['birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx']='/usr/local/birensupa/sdk/1.11.0.1.rc1'
  ['biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107']='/usr/local/birensupa/sdk/1.10.0.0-24145.lkg'
)
for IMG in 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'; do
  SDK="${LIBDIR[$IMG]}"
  echo ""; echo "############ $IMG ############"
  docker run --rm --entrypoint bash -e "BR_SDK_DIR=$SDK" "$IMG" -c '
S="$BR_SDK_DIR"
LP=$(find "$S" -name "*.so*" 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -)
export LD_LIBRARY_PATH="$LP"
python3 -c "
import torch, torch_br
print(\"torch:\", torch.__version__)
print(\"torch_br:\", getattr(torch_br, \"__version__\", \"?\"), torch_br.__file__)
for ns in [\"biren\",\"supa\",\"br\",\"cuda\",\"xpu\",\"npu\",\"mlu\"]:
    if hasattr(torch, ns):
        m = getattr(torch, ns)
        try: extra = \"device_count=%s\" % m.device_count()
        except Exception as e: extra = \"dc_err:%s\" % type(e).__name__
        print(\"torch.%-6s %s\" % (ns, extra))
import torch._C as C
t = C._dispatch_dump_table(\"aten::add.Tensor\")
print(\"add.Tensor PrivateUse1:\", [l.strip()[:90] for l in t.splitlines() if \"PrivateUse1\" in l][:1])
"'
done
echo "=== END probe7 ==="
