#!/bin/bash
# P800 · 停止任务前的只读盘点（2026-10-14）
# 纪律：本脚本**只读**，不动任何容器/进程；目的是把「我的」与「他人的」分开。
set -u

echo "===== [0] 时间/主机/负载 ====="
date "+[%F %T]"; hostname; uptime | head -1

echo
echo "===== [1] 全部容器（含 stopped）====="
docker ps -a --format '{{.Names}}\t{{.Status}}\t{{.RunningFor}}\t{{.Image}}' | sort

echo
echo "===== [2] 与我相关的容器（前缀 hliu553 / dc-）====="
docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E "hliu553|^dc-" || echo "  （无）"

echo
echo "===== [3] 每个在跑容器里有没有**计算进程**（非 idle）====="
for c in $(docker ps --format '{{.Names}}'); do
    n=$(docker top "$c" 2>/dev/null | grep -ciE "python|torch|vllm|EngineCore|train|infer|runner" || true)
    printf "  %-34s 计算类进程行数=%s\n" "$c" "$n"
done

echo
echo "===== [4] 我自己的容器内部进程明细 ====="
for c in $(docker ps --format '{{.Names}}' | grep -E "hliu553"); do
    echo "  --- $c ---"
    docker top "$c" 2>/dev/null | sed 's/^/    /' | head -15
done

echo
echo "===== [5] 宿主上与我相关的进程（路径含 /data2/hliu553 或用户名 hliu553）====="
ps -eo pid,user,etime,stat,cmd --sort=start_time \
  | grep -E "/data2/hliu553|[h]liu553|serve_standard|[v]llm serve" \
  | grep -vE "docker top|grep" | head -30
echo "  （以上若为空 = 宿主侧无我方进程）"

echo
echo "===== [6] 我的数据目录是否被进程占用 ====="
command -v fuser >/dev/null && { fuser -v /data2/hliu553 2>&1 | head -10; } || echo "  （无 fuser）"
echo "  目录规模："; du -sh /data2/hliu553 2>/dev/null | head -3

echo
echo "===== [7] 设备占用（xpu-smi，裸命令）====="
xpu-smi 2>&1 | head -40

echo
echo "===== [8] 我方的容器挂载集（用于判断停它会不会碰到别人的卡）====="
for c in $(docker ps --format '{{.Names}}' | grep -E "hliu553"); do
    printf "  %-34s devices=" "$c"
    docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" 2>/dev/null
done

echo
echo "===== [9] 其他人在跑的容器（**只读，不动**）====="
docker ps --format '{{.Names}}\t{{.Status}}' | grep -vE "hliu553|^dc-" | head -20

echo
echo "===== 盘点结束 $(date '+%F %T') ====="
