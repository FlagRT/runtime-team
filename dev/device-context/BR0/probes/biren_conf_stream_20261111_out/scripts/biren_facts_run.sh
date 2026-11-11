#!/usr/bin/env bash
# 壁仞 BR_0：S12 流优先级事实 + i6 重叠率随工作量变化（归档用，重跑可复现）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
OUT=/tmp/biren_facts_$RUNID
mkdir -p "$OUT"
LOG="$OUT/facts_s12_i6_$RUNID.log"

cp /tmp/pp.py /tmp/pp2.py /tmp/ov_scale.py "$OUT"/ 2>/dev/null
docker cp /tmp/pp.py "$C":/tmp/pp.py >/dev/null 2>&1
docker cp /tmp/pp2.py "$C":/tmp/pp2.py >/dev/null 2>&1
docker cp /tmp/ov_scale.py "$C":/tmp/ov_scale.py >/dev/null 2>&1

{
  echo "=== 壁仞 BR_0 事实归档（S12 流优先级 / i6 重叠率）==="
  echo "[HOST] $(hostname) $(date '+%F %T %Z')"
  echo "[IMG ] $(docker inspect "$C" --format '{{.Config.Image}}')"
  echo "[VIS ] $(docker inspect "$C" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^BIREN_VISIBLE_DEVICES=' || echo 'n/a')"
  echo "[NODE] $(docker exec "$C" bash -c 'ls /dev/biren/ | tr "\n" " "')"
  echo
  echo "--- A. 流优先级：范围 API / 构造 / 回读一致性 ---"
  docker exec "$C" bash -c ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 /tmp/pp.py"
  echo
  echo "--- A2. 档内逐值设置-回读（含 range 调用结果）---"
  docker exec "$C" bash -c ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 /tmp/pp2.py; echo rc=\$?"
  echo
  echo "--- B. i6 口径随工作量变化（预热后计时）---"
  docker exec "$C" bash -c ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 /tmp/ov_scale.py"
  echo
  echo "=== END ==="
} 2>&1 | grep -viE '^\s*warn|UserWarning|warnings.warn' | tee "$LOG"

cp /tmp/pp.py /tmp/pp2.py /tmp/ov_scale.py "$OUT"/ 2>/dev/null
echo "OUT=$OUT"
