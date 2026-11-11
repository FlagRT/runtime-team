#!/usr/bin/env bash
# MLU590 兼容性核对 · Step1 宿主驱动溯源 + L1 容器版本（不挂设备、不占卡、只读）
# 用法：ssh <Mlu-1|Mlu-2> 'bash -s' < /tmp/mlu_compat_step1.sh
set -u
TAG='harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0'
WD=/tmp/mlu_compat_20261105
mkdir -p "$WD"

echo "=== SELF_CHECK mlu_compat_step1 2026-11-05 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- cnmon header ---'; cnmon 2>&1 | head -6
echo '--- cnmon binary / owner pkg ---'; ls -l /usr/bin/cnmon 2>&1; dpkg -S /usr/bin/cnmon 2>&1 | head -4
echo '--- driver-related dpkg packages ---'; dpkg -l 2>/dev/null | grep -iE 'cambricon|neuware|cnmon|mlu|cntoolkit' | head -25
echo '--- dpkg.log driver install/upgrade tail ---'; zgrep -h -iE 'cambricon|neuware|cnmon|cntoolkit' /var/log/dpkg.log* 2>/dev/null | tail -25
echo '--- kernel module version ---'; cat /sys/module/cambricon_drv/version 2>&1 | head -3; ls -ld /sys/module/cambricon_drv 2>&1
echo '--- dev nodes ---'; ls -l /dev/cambricon* 2>&1 | head -12
echo '--- image inspect (locked runtime image) ---'; docker inspect "$TAG" --format 'ID={{.Id}} Created={{.Created}}' 2>&1
echo '--- our own containers ---'; docker ps -a --filter 'name=dc-mlu590' --format '{{.Names}}|{{.Image}}|{{.Status}}' 2>&1

cat > "$WD/l1_probe.py" <<'PY'
import sys, json, subprocess
print("=== L1 CONTAINER SELF-CHECK ===")
print("os-release:", [l for l in open("/etc/os-release").read().splitlines() if l.startswith(("NAME=", "VERSION=", "PRETTY_NAME=", "ID=", "VERSION_ID="))])
print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch IMPORT_FAIL:", type(e).__name__, e)
try:
    import torch_mlu
    print("torch_mlu:", torch_mlu.__version__)
    print("device_count(no --device):", torch.mlu.device_count())
except Exception as e:
    print("torch_mlu IMPORT_FAIL:", type(e).__name__, e)
try:
    import triton
    print("triton:", triton.__version__)
except Exception as e:
    print("triton IMPORT_FAIL:", type(e).__name__, e)
try:
    import flag_gems
    print("flag_gems:", getattr(flag_gems, "__version__", "no __version__"), flag_gems.__file__)
except Exception as e:
    print("flag_gems IMPORT_FAIL:", type(e).__name__, e)
PY

echo '--- L1: container OS / python / imports (docker run --rm, NO device) ---'
docker run --rm --network none -v "$WD/l1_probe.py:/l1_probe.py:ro" --entrypoint /flagos/bin/python3 "$TAG" /l1_probe.py 2>&1 | head -40
echo '--- L1: pip list filtered ---'
docker run --rm --network none --entrypoint bash "$TAG" -lc '/flagos/bin/python3 -m pip list 2>/dev/null | grep -iE "^(torch|torch_mlu|torch_mlu_ops|triton|flag_gems|flagtree|numpy|transformers|vllm) "' 2>&1 | head -30
echo '--- L1: neuware version / package versions ---'
docker run --rm --network none --entrypoint bash "$TAG" -lc 'cat /usr/local/neuware/version.txt 2>&1; echo "-- env TOOLKIT/PKGS --"; env | grep -iE "TOOLKIT|PKGS|NEUWARE" | head -10' 2>&1 | head -25
echo "=== END step1 ==="
