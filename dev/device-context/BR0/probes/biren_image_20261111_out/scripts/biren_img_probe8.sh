#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe8 2026-11-11 ==="
for IMG in 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'; do
  echo ""; echo "############ $IMG ############"
  docker run --rm --entrypoint bash "$IMG" -c '
echo "--- libbesu everywhere ---"; find / -name "libbesu*" 2>/dev/null | head -6
echo "--- /usr/lib/libbesu* size/links ---"; ls -la /usr/lib/libbesu* 2>/dev/null | head -4
echo "--- ldconfig cache has besu? ---"; ldconfig -p 2>/dev/null | grep -i besu | head -3
' 2>&1 | head -18
done
echo "=== END probe8 ==="
