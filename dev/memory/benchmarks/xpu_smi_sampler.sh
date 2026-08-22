#!/usr/bin/env bash
# P800 宿主侧 HBM/利用率采样器 (交叉验证用, 对应 910c 的 npu-smi 采样方法学)
# 用法: bash benchmarks/xpu_smi_sampler.sh <card> <interval_s> <seconds> [out.csv]
# 输出: ts,card,used_mib,total_mib,util_pct
set -u
CARD=${1:-1}
INTERVAL=${2:-2}
SECONDS=${3:-120}
OUT=${4:-/data2/xliu969/code/runtime-team/dev/memory/benchmarks/out/xpu_smi_${CARD}.csv}
mkdir -p "$(dirname "$OUT")"
echo "ts,card,used_mib,total_mib,util_pct" > "$OUT"
END=$(( $(date +%s) + SECONDS ))
while [ "$(date +%s)" -lt "$END" ]; do
  RAW=$(xpu-smi -i "$CARD" -q 2>/dev/null)
  USED=$(printf '%s' "$RAW" | awk '/Used/{print $3; exit}' | tr -d 'MiB')
  TOTAL=$(printf '%s' "$RAW" | awk '/Total/{print $3; exit}' | tr -d 'MiB')
  UTIL=$(printf '%s' "$RAW" | awk '/Xpu/{print $3; exit}' | tr -d '%')
  echo "$(date +%s),$CARD,${USED:-NA},${TOTAL:-NA},${UTIL:-NA}" >> "$OUT"
  sleep "$INTERVAL"
done
echo "done: $OUT"
