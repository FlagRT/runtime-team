#!/usr/bin/env bash
# 壁仞 BR_0 资源普查 · Step1（只读：brsmi 卡状态 / SDK 内容 / sudo 能力验证）
set -u
echo "=== SELF_CHECK biren_step1 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- brsmi help (first lines) ---'; brsmi --help 2>&1 | head -20
echo '--- brsmi default dump ---'; brsmi 2>&1 | head -60
echo '--- brsmi -l / list ---'; brsmi -l 2>&1 | head -30 || true
echo '--- /dev/biren* detail ---'; ls -l /dev/biren* 2>/dev/null
echo '--- SDK top ---'; ls /usr/local/birensupa/ 2>/dev/null
echo '--- SDK version files ---'; find /usr/local/birensupa -maxdepth 2 -iname '*version*' -o -maxdepth 2 -iname '*.version' 2>/dev/null | head -6; cat /usr/local/birensupa/version* 2>/dev/null | head -5
echo '--- sudo capability (non-interactive test, rc only) ---'; echo 'sudo-nopasswd-test' >/dev/null; sudo -n true 2>/dev/null && echo 'SUDO_NOPASSWD' || echo 'SUDO_NEEDS_PASSWORD'
echo '--- home dir ---'; ls ~ | head -10; df -h ~ | tail -1
echo '--- network / proxy ---'; curl -sS -m 8 -o /dev/null -w 'github=%{http_code}\n' https://github.com 2>&1 | head -2
echo "=== END step1 ==="
