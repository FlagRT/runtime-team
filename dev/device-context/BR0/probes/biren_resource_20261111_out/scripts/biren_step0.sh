#!/usr/bin/env bash
# 壁仞 BR_0 资源普查 · Step0（只读，不改任何配置、不占卡、不动他人容器）
set -u
echo "=== SELF_CHECK biren_step0 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- whoami / id ---'; whoami; id
echo '--- sudo? ---'; sudo -n true 2>&1 | head -2; grep -E '^(sudo|docker)' /etc/group | head -5
echo '--- os-release ---'; grep -E '^(NAME|VERSION|PRETTY_NAME|ID|VERSION_ID)=' /etc/os-release
echo '--- kernel ---'; uname -a
echo '--- cpu / mem ---'; nproc; free -h | head -2
echo '--- disk ---'; df -h | grep -vE 'tmpfs|loop|overlay$' | head -10
echo '--- nvidia-smi-like vendor tools ---'; for t in biren-smi smi bri-smi tops-smi; do command -v $t >/dev/null 2>&1 && echo "FOUND: $t"; done; ls /usr/bin/ | grep -iE 'smi|biren|tops' | head -10
echo '--- lspci accelerator ---'; lspci 2>/dev/null | grep -iE 'biren|accelerat|processor.*device|100[a-f0-9]:' | head -10
echo '--- /dev vendor nodes ---'; ls /dev/ | grep -iE 'biren|br|tops|davinci|cambricon|xpu|npu' | head -15
echo '--- driver modules ---'; lsmod | grep -iE 'biren|br|tops' | head -10
echo '--- module versions ---'; for m in $(lsmod | grep -iE 'biren|tops' | awk '{print $1}' | sort -u | head -5); do echo "[$m] $(cat /sys/module/$m/version 2>/dev/null || echo no-version-file)"; done
echo '--- docker ---'; docker --version 2>&1 | head -1; docker ps -a --format '{{.Names}}|{{.Image}}|{{.Status}}' 2>&1 | head -15
echo '--- docker images ---'; docker images --format '{{.Repository}}:{{.Tag}}|{{.ID}}|{{.CreatedSince}}' 2>&1 | grep -iE 'biren|br|flagos|pytorch' | head -15
echo '--- SDK paths ---'; ls -ld /usr/local/biren* /usr/local/BR* /opt/biren* /opt/BR* 2>/dev/null | head -8
echo '--- env hints ---'; env | grep -iE 'biren|BR_' | head -10
echo "=== END step0 ==="
