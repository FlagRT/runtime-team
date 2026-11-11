#!/bin/bash
# P800 · 停止我方任务（2026-10-14，用户要求：停止任务以便迁移数据）
# 纪律：① 只停**我方**容器（前缀 hliu553）；② 停前/停后对**他人容器**逐一比对，证明未扰动；
#       ③ 不删任何容器、不删任何数据（迁移前保持可复现）。
set -u

MY=hliu553-device-context-p800
OUT=${OUT:-/data2/hliu553/stop_tasks_20261014}
mkdir -p "$OUT"

echo "===== [0] 开始 $(date '+%F %T') ====="
echo "宿主: $(hostname) | 负载: $(uptime | sed 's/.*load average: //')"

echo
echo "===== [1] 停止前：全容器状态快照（用于停后比对）====="
docker ps -a --format '{{.Names}}|{{.Status}}' | sort > "$OUT/containers_before.txt"
wc -l < "$OUT/containers_before.txt" | sed 's/^/  容器总数=/' 
echo "  --- 我方容器 ---"
grep -E "^hliu553" "$OUT/containers_before.txt" | sed 's/^/    /'

echo
echo "===== [2] 停止前：我方容器内是否真的没有计算进程 ====="
docker top "$MY" 2>/dev/null | sed 's/^/    /'
n=$(docker top "$MY" 2>/dev/null | grep -ciE "python|torch|vllm|EngineCore|train|infer|runner" || true)
echo "  计算类进程数=$n（0 = 空转，停它不会打断任何取数）"

echo
echo "===== [3] 执行停止：docker stop $MY ====="
docker stop "$MY" 2>&1 | sed 's/^/  /'
sleep 3
echo "  停后状态: $(docker inspect -f '{{.State.Status}} (exit={{.State.ExitCode}})' "$MY")"

echo
echo "===== [4] 停止后：他人容器是否被扰动（逐一比对）====="
docker ps -a --format '{{.Names}}|{{.Status}}' | sort > "$OUT/containers_after.txt"
# 只比「他人容器」的**运行状态**（名字前缀 hliu553 之外的）
grep -vE "^hliu553" "$OUT/containers_before.txt" | sed 's/|.*|/|/' > "$OUT/others_before.txt"
grep -vE "^hliu553" "$OUT/containers_after.txt"  | sed 's/|.*|/|/' > "$OUT/others_after.txt"
if diff -u "$OUT/others_before.txt" "$OUT/others_after.txt" > "$OUT/others_diff.txt"; then
    echo "  ✅ 他人容器状态：逐一相同（未扰动，共 $(wc -l < "$OUT/others_before.txt" | tr -d ' ') 个）"
else
    echo "  ⚠️ 他人容器状态有变化（可能是他人自己的编排动作，非我方造成）——差异如下："
    sed 's/^/    /' "$OUT/others_diff.txt" | head -20
fi

echo
echo "===== [5] 我方容器最终状态 ====="
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E "^hliu553" | sed 's/^/  /'

echo
echo "===== [6] 宿主残留进程复查（我方）====="
ps -eo pid,user,etime,cmd | grep -E "/data2/hliu553|serve_standard|[v]llm serve" | grep -vE "grep|bash -s" | head -10
echo "  （空 = 无残留）"

echo
echo "===== [7] 设备占用（停后）====="
xpu-smi 2>&1 | grep -E "^\| +[0-9]|MiB / 98304MiB" | paste - - 2>/dev/null | head -8

echo
echo "===== [8] 我方数据足迹（迁移用清单）====="
du -sh /data2/hliu553 2>/dev/null | sed 's/^/  合计: /'
du -sh /data2/hliu553/* 2>/dev/null | sort -h | sed 's/^/    /'
echo "  /home/hliu553: $(du -sh /home/hliu553 2>/dev/null | awk '{print $1}')"
echo "  我方 /tmp 遗留: $(find /tmp -maxdepth 1 -user hliu553 2>/dev/null | wc -l | tr -d ' ') 项"

echo
echo "===== 结束 $(date '+%F %T') ====="
echo "证据目录: $OUT"
