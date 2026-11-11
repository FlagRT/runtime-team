#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe6 2026-11-11 ==="
declare -A LIBDIR=(
  ['birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx']='/usr/local/birensupa/sdk/1.11.0.1.rc1'
  ['biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107']='/usr/local/birensupa/sdk/1.10.0.0-24145.lkg'
)
for IMG in "${!LIBDIR[@]}"; do
  SDK="${LIBDIR[$IMG]}"
  echo ""; echo "############ $IMG ############"
  echo "SDK dir: $SDK"
  docker run --rm --entrypoint bash -e "BR_SDK_DIR=$SDK" "$IMG" -c '
S="$BR_SDK_DIR"; P="$(pip show torch-br 2>/dev/null | grep ^Location | cut -d" " -f2)/torch_br"
LIBS=$(find "$S" -name "*.so*" -path "*lib*" 2>/dev/null | xargs -n1 dirname | sort -u | head -6)
export LD_LIBRARY_PATH="$LIBS:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | head -c 300; echo
echo "torch_br path: $P"
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
"'
done
echo "=== END probe6 ==="
