#!/usr/bin/env bash
# MLU590 兼容性核对 · Step0 宿主/容器只读探查（不占卡、不改任何资产）
# 用法：ssh Mlu-1 'bash -s' < /tmp/mlu_compat_step0.sh
set -u
echo "=== SELF_CHECK mlu_compat_step0 2026-11-05 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo "--- whoami / id ---"; whoami; id
echo "--- os-release ---"; grep -E '^(NAME|VERSION|PRETTY_NAME|ID|VERSION_ID)=' /etc/os-release
echo "--- kernel ---"; uname -a
echo "--- docker version ---"; docker --version 2>&1 | head -3
echo "--- docker ps -a ---"; docker ps -a --format '{{.Names}}|{{.Image}}|{{.Status}}' 2>&1 | head -40
echo "--- docker images (cambricon/neuware/mlu) ---"; docker images --format '{{.Repository}}:{{.Tag}}|{{.ID}}|{{.CreatedSince}}' 2>&1 | grep -iE 'cambricon|neuware|mlu' | head -30
echo "--- cnmon (first 100 lines) ---"; cnmon 2>&1 | head -100
echo "--- neuware host path ---"; ls -ld /usr/local/neuware 2>&1
echo "--- disk ---"; df -h / /srv 2>&1 | head -8
echo "=== END step0 ==="
