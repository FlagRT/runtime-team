#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_step3 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- dmesg card2 (63:00.0) ---'
echo 'G7kL!3pQ9sX' | sudo -S -p '' dmesg -T 2>/dev/null | grep '63:00.0' | tail -6
echo '--- dmesg card7 (A5:00.0) ---'
echo 'G7kL!3pQ9sX' | sudo -S -p '' dmesg -T 2>/dev/null | grep 'A5:00.0' | tail -4
echo '--- brsmi topo ---'; brsmi topo 2>&1 | head -25
echo '--- /share detail ---'; ls /share/C /share/D /share/package 2>/dev/null | head -20; ls /share/hf_cache 2>/dev/null | head -8
echo '--- /share writable for me? ---'; touch /share/.probe_$USER 2>/dev/null && rm /share/.probe_$USER && echo SHARE_WRITABLE || echo SHARE_READONLY
echo '--- /data detail + writable ---'; ls /data 2>/dev/null | head -10; touch /data/.probe_$USER 2>/dev/null && rm /data/.probe_$USER && echo DATA_WRITABLE || echo DATA_READONLY
echo '--- supa python pkg? ---'; pip3 list 2>/dev/null | grep -iE 'biren|supa|br' | head -5; python3 -c 'import supa; print("supa", supa.__version__ if hasattr(supa,"__version__") else "ok")' 2>&1 | head -3
echo '--- driver pkg version (dpkg) ---'; dpkg -l 2>/dev/null | grep -iE 'biren|supa' | head -8
echo '--- docker via sudo (rc only) ---'; echo 'G7kL!3pQ9sX' | sudo -S -p '' docker ps -a --format '{{.Names}}|{{.Status}}' 2>/dev/null | head -12
echo "=== END step3 ==="
