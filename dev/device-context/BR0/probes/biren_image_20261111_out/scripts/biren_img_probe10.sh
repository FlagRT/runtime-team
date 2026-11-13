#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe10 2026-11-11 ==="
for IMG in 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'; do
  echo ""; echo "############ $IMG ############"
  docker run --rm --entrypoint bash "$IMG" -c '
echo "--- supa*.so / biren*.so in SDK ---"
find /usr/local/birensupa -name "*.so*" 2>/dev/null | grep -iE "supa|biren|br" | head -12
echo "--- torch_br bundled libs ---"
TB=$(pip show torch-br 2>/dev/null | grep ^Location | cut -d" " -f2)/torch_br
find "$TB" -name "*.so*" 2>/dev/null | head -8
echo "--- ldd torch_br._C ---"
LP=$(find /usr/local/birensupa -name "*.so*" 2>/dev/null | xargs -n1 dirname 2>/dev/null | sort -u | paste -sd: -)
export LD_LIBRARY_PATH="$LP"
ldd "$TB/_C.cpython-310-x86_64-linux-gnu.so" 2>/dev/null | grep -E "not found" | head -8
' 2>&1 | head -30
done
echo "=== END probe10 ==="
