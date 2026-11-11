#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe11 2026-11-11 ==="
declare -A LIBDIR=(
  ['birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx']='/usr/local/birensupa/sdk/1.11.0.1.rc1'
  ['biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107']='/usr/local/birensupa/1.10.0.0-24145.lkg'
)
IMG1='birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx'
IMG2='biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'
probe() {
  local IMG="$1" SDK="$2"
  echo ""; echo "############ $IMG (sdk=$SDK) ############"
  docker run --rm --entrypoint bash -e "BR_SDK_DIR=$SDK" "$IMG" -c '
S="$BR_SDK_DIR"
TB=$(pip show torch-br 2>/dev/null | grep ^Location | cut -d" " -f2)/torch_br
TOL=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))" 2>/dev/null || python3 -c "import sysconfig;print(sysconfig.get_paths()[\"purelib\"])")
LIB=$(find "$TB/lib" "$S" -name "*.so*" 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -)
TorchLib=$(python3 - <<PY
import glob,os
cands = glob.glob("/usr/local/lib/python3*/dist-packages/torch/lib") + glob.glob("/usr/lib/python3*/dist-packages/torch/lib")
print(cands[0] if cands else "")
PY
)
export LD_LIBRARY_PATH="$TorchLib:$LIB:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH(first 200): ${LD_LIBRARY_PATH:0:200}"
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
for op in ["aten::add.Tensor","aten::mm"]:
    t = C._dispatch_dump_table(op)
    print(op, \"PU1:\", [l.strip()[:95] for l in t.splitlines() if \"PrivateUse1\" in l][:1])
"'
}
probe "$IMG1" '/usr/local/birensupa/sdk/1.11.0.1.rc1'
probe "$IMG2" '/usr/local/birensupa/sdk/1.10.0.0-24145.lkg'
echo "=== END probe11 ==="
