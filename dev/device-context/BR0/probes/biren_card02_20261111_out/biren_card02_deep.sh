#!/usr/bin/env bash
# 壁仞 BR_0 · 卡 0/2 异常专项复核（只读；供上游报告的原文证据采集）
set -u
echo "=== SELF_CHECK biren_card02_deepcheck 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
PW='G7kL!3pQ9sX'

echo '--- [1] brsmi full dump (now) ---'
brsmi 2>&1

echo '--- [2] brsmi query-gpu per card (index/name/temp/pwr/mem/util) ---'
brsmi gpu --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv 2>&1 | head -15

echo '--- [3] card_0 (03:00.0) FULL dmesg timeline ---'
echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | grep '03:00.0' | head -60

echo '--- [4] card_2 (63:00.0) FULL dmesg timeline ---'
echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | grep '63:00.0' | head -60

echo '--- [5] boot time / kernel log start (ring buffer coverage) ---'
uptime -s 2>/dev/null; echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | head -2

echo '--- [6] earlier occurrences? (count ERR/reset per bus) ---'
echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | grep -iE 'biren' | grep -cE 'ERR' | xargs echo "total biren ERR lines:"
echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | grep -iE 'hot reset' | awk '{print $NF}' | sort | uniq -c | head -5
echo '--- per-bus ERR/reset counts ---'
for bus in 03:00.0 05:00.0 63:00.0 65:00.0 83:00.0 85:00.0 A3:00.0 A5:00.0; do
  errs=$(echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | grep "$bus" | grep -cE '\[ERR|fail|hang|reset')
  total=$(echo "$PW" | sudo -S -p '' dmesg -T 2>/dev/null | grep -c "$bus")
  echo "bus $bus: total_lines=$total err_or_reset_lines=$errs"
done

echo '--- [7] syslog/journal around 11-10 16:00-17:30 (card events) ---'
echo "$PW" | sudo -S -p '' journalctl --since '2026-11-10 15:50' --until '2026-11-10 17:40' -k --no-pager 2>/dev/null | grep -iE 'biren|03:00.0|63:00.0' | grep -vE 'is_queue_busy' | head -40

echo '--- [8] card_0 residual memory recheck ---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>&1 | head -10

echo '--- [9] PCI link status of card 0/2 (via sudo lspci) ---'
echo "$PW" | sudo -S -p '' lspci -s 03:00.0 -vvv 2>/dev/null | grep -iE 'LNKSta|LNKCap|Status:' | head -6
echo '---'
echo "$PW" | sudo -S -p '' lspci -s 63:00.0 -vvv 2>/dev/null | grep -iE 'LNKSta|LNKCap|Status:' | head -6

echo '--- [10] brsmi drain/reset help (info only, NOT executing) ---'
brsmi drain --help 2>&1 | head -12
brsmi reset --help 2>&1 | head -12

echo "=== END deepcheck ==="
