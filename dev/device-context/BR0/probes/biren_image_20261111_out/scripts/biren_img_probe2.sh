#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_image_probe2 2026-11-11 ==="
IMG1='birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx'
IMG2='biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-cmake-3.24.1-20261107'
echo '--- [A] img1 config (Env/Entrypoint/Cmd) ---'
docker inspect "$IMG1" --format '{{json .Config.Env}}' | tr ',' '\n' | grep -viE 'path=|lang=|lc_' | head -15
docker inspect "$IMG1" --format 'ENTRYPOINT={{json .Config.Entrypoint}} CMD={{json .Config.Cmd}}'
echo '--- [B] img1 real env: pip list + python paths ---'
docker run --rm --entrypoint bash "$IMG1" -lc 'pip list 2>/dev/null | grep -iE "supa|biren|torch|triton" | head -12; echo ---; ls /opt/ 2>/dev/null; ls /usr/local/ | head -10; which python3 pip; python3 -c "import torch; print(torch.__version__)"' 2>&1 | head -25
echo ''
echo '--- [C] img2 probe (correct tag) ---'
docker run --rm -v /tmp/img_probe.py:/img_probe.py:ro --entrypoint python3 "$IMG2" /img_probe.py 2>&1 | head -20 || \
docker run --rm --entrypoint bash "$IMG2" -lc 'cat /img_probe.py 2>/dev/null || true; pip list 2>/dev/null | grep -iE "supa|biren|torch" | head -8' 2>&1 | head -20
echo '--- [D] img2 entrypoint fallback: env/pip ---'
docker inspect "$IMG2" --format 'ENTRYPOINT={{json .Config.Entrypoint}} CMD={{json .Config.Cmd}}'
docker run --rm --entrypoint bash "$IMG2" -lc 'pip list 2>/dev/null | grep -iE "supa|biren|torch" | head -10; ls /opt/ 2>/dev/null | head -8' 2>&1 | head -18
echo "=== END probe2 ==="
