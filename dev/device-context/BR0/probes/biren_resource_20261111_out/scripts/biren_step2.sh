#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_step2 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- sudo with password (rc only, no privileged op beyond id) ---'
echo 'G7kL!3pQ9sX' | sudo -S -p '' id >/dev/null 2>&1 && echo 'SUDO_PASSWORD_OK' || echo 'SUDO_PASSWORD_FAIL'
echo '--- dmesg biren errors (via sudo) ---'
echo 'G7kL!3pQ9sX' | sudo -S -p '' dmesg -T 2>/dev/null | grep -iE 'biren|card' | grep -iE 'error|fail|reset|timeout|xid' | tail -15
echo '--- dmesg biren recent (any) ---'
echo 'G7kL!3pQ9sX' | sudo -S -p '' dmesg -T 2>/dev/null | grep -i 'biren' | tail -8
echo '--- container-toolkit contents ---'
ls /usr/local/birensupa/container-toolkit/ 2>/dev/null | head -10
echo '--- driver contents ---'
ls /usr/local/birensupa/driver/ 2>/dev/null | head -10
echo '--- sudcgm contents ---'
ls /usr/local/birensupa/sudcgm/ 2>/dev/null | head -8
echo '--- /share top (shared assets?) ---'
ls /share/ 2>/dev/null | head -15
echo '--- home writable + current usage ---'
touch ~/.biren_probe_test 2>/dev/null && rm ~/.biren_probe_test && echo 'HOME_WRITABLE'
echo '--- python3 ---'; python3 --version 2>&1
echo '--- docker sock group ---'; ls -l /var/run/docker.sock 2>/dev/null
echo "=== END step2 ==="
