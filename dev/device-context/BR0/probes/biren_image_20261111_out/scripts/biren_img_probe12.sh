#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe12 2026-11-11 ==="
echo "aW1wb3J0IHRvcmNoLCB0b3JjaF9icgpwcmludCgidG9yY2g6IiwgdG9yY2guX192ZXJzaW9uX18pCnByaW50KCJ0b3JjaF9icjoiLCBnZXRhdHRyKHRvcmNoX2JyLCAiX192ZXJzaW9uX18iLCAiPyIpLCB0b3JjaF9ici5fX2ZpbGVfXykKZm9yIG5zIGluIFsiYmlyZW4iLCJzdXBhIiwiYnIiLCJjdWRhIiwieHB1IiwibnB1IiwibWx1Il06CiAgICBpZiBoYXNhdHRyKHRvcmNoLCBucyk6CiAgICAgICAgbSA9IGdldGF0dHIodG9yY2gsIG5zKQogICAgICAgIHRyeTogZXh0cmEgPSAiZGV2aWNlX2NvdW50PSVzIiAlIG0uZGV2aWNlX2NvdW50KCkKICAgICAgICBleGNlcHQgRXhjZXB0aW9uIGFzIGU6IGV4dHJhID0gImRjX2VycjolcyIgJSB0eXBlKGUpLl9fbmFtZV9fCiAgICAgICAgcHJpbnQoInRvcmNoLiUtNnMgJXMiICUgKG5zLCBleHRyYSkpCmltcG9ydCB0b3JjaC5fQyBhcyBDCmZvciBvcCBpbiBbImF0ZW46OmFkZC5UZW5zb3IiLCJhdGVuOjptbSJdOgogICAgdCA9IEMuX2Rpc3BhdGNoX2R1bXBfdGFibGUob3ApCiAgICBwcmludChvcCwgIlBVMToiLCBbbC5zdHJpcCgpWzo5NV0gZm9yIGwgaW4gdC5zcGxpdGxpbmVzKCkgaWYgIlByaXZhdGVVc2UxIiBpbiBsXVs6MV0pCg==" | base64 -d > /tmp/ns_probe.py
probe() {
  local IMG="$1" SDK="$2"
  echo ""; echo "############ $IMG (sdk=$SDK) ############"
  docker run --rm -v /tmp/ns_probe.py:/ns_probe.py:ro --entrypoint bash -e "BR_SDK_DIR=$SDK" "$IMG" -c '
S="$BR_SDK_DIR"
TB=$(pip show torch-br 2>/dev/null | grep ^Location | cut -d" " -f2)/torch_br
TorchLib=$(ls -d /usr/local/lib/python3*/dist-packages/torch/lib 2>/dev/null | head -1)
LIB=$(find "$TB/lib" "$S" -name "*.so*" 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -)
export LD_LIBRARY_PATH="$TorchLib:$LIB"
python3 /ns_probe.py'
}
probe 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' '/usr/local/birensupa/sdk/1.11.0.1.rc1'
probe 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107' '/usr/local/birensupa/sdk/1.10.0.0-24145.lkg'
rm -f /tmp/ns_probe.py
echo "=== END probe12 ==="
