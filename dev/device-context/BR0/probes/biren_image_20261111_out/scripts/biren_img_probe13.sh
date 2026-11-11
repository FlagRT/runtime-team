#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe13 2026-11-11 ==="
echo "aW1wb3J0IHRvcmNoLCB0b3JjaF9iciwgb3MKcHJpbnQoInRvcmNoOiIsIHRvcmNoLl9fdmVyc2lvbl9fKQpwcmludCgidG9yY2hfYnI6IiwgZ2V0YXR0cih0b3JjaF9iciwgIl9fdmVyc2lvbl9fIiwgIj8iKSwgdG9yY2hfYnIuX19maWxlX18pCmZvciBucyBpbiBbImJpcmVuIiwic3VwYSIsImJyIiwiY3VkYSIsInhwdSIsIm5wdSIsIm1sdSJdOgogICAgaWYgaGFzYXR0cih0b3JjaCwgbnMpOgogICAgICAgIG0gPSBnZXRhdHRyKHRvcmNoLCBucykKICAgICAgICB0cnk6IGV4dHJhID0gImRldmljZV9jb3VudD0lcyIgJSBtLmRldmljZV9jb3VudCgpCiAgICAgICAgZXhjZXB0IEV4Y2VwdGlvbiBhcyBlOiBleHRyYSA9ICJkY19lcnI6JXMiICUgdHlwZShlKS5fX25hbWVfXwogICAgICAgIHByaW50KCJ0b3JjaC4lLTZzICVzIiAlIChucywgZXh0cmEpKQppbXBvcnQgdG9yY2guX0MgYXMgQwpmb3Igb3AgaW4gWyJhdGVuOjphZGQuVGVuc29yIiwiYXRlbjo6bW0iXToKICAgIHQgPSBDLl9kaXNwYXRjaF9kdW1wX3RhYmxlKG9wKQogICAgcHJpbnQob3AsICJQVTE6IiwgW2wuc3RyaXAoKVs6OTVdIGZvciBsIGluIHQuc3BsaXRsaW5lcygpIGlmICJQcml2YXRlVXNlMSIgaW4gbF1bOjFdKQo=" | base64 -d > /tmp/ns_probe2.py
probe() {
  local IMG="$1" SDK="$2"
  echo ""; echo "############ $IMG (sdk=$SDK) ############"
  docker run --rm -v /tmp/ns_probe2.py:/ns_probe2.py:ro --entrypoint bash -e "BR_SDK_DIR=$SDK" "$IMG" -c '
S="$BR_SDK_DIR"
TB=$(pip show torch-br 2>/dev/null | grep ^Location | cut -d" " -f2)/torch_br
TorchLib=$(ls -d /usr/local/lib/python3*/dist-packages/torch/lib 2>/dev/null | head -1)
LIB=$(find "$TB/lib" "$S" -name "*.so*" 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -)
# 关键修正：brbpp 提供 libbesu —— 用它替换 /usr/lib 里的坏库（LD_PRELOAD 只收第一个）
BESU=$(ls "$S"/brbpp/lib*/libbesu.so.1* 2>/dev/null | sort -V | tail -1)
echo "besu pick: $BESU"
export LD_LIBRARY_PATH="$TorchLib:$LIB"
export LD_PRELOAD="$BESU"
python3 /ns_probe2.py'
}
probe 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' '/usr/local/birensupa/sdk/1.11.0.1.rc1'
probe 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107' '/usr/local/birensupa/sdk/1.10.0.0-24145.lkg'
rm -f /tmp/ns_probe2.py
echo "=== END probe13 ==="
