#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe5 2026-11-11 ==="
for IMG in 'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' 'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'; do
  echo ""; echo "############ $IMG ############"
  docker run --rm --entrypoint bash "$IMG" -c '
find / -name "libsupti.so*" 2>/dev/null | head -3
echo "--- entrypoint.d real ---"; find /usr/local/birensupa -name "entrypoint.d" -type d 2>/dev/null | head -2
ED=$(find /usr/local/birensupa -name "entrypoint.d" -type d 2>/dev/null | head -1)
[ -n "$ED" ] && ls "$ED" | head -6 && echo "--- one script env lines ---" && grep -hE "export" "$ED"/*.sh 2>/dev/null | head -10
' 2>&1 | head -22
done
echo "=== END probe5 ==="
