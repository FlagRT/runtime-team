#!/usr/bin/env bash
# 壁仞 BR_0：conformance 13 例 + 推理 6 例（真机，卡 85:00.0 / a5:00.0）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_conf_$RUNID
CB=/work/out_biren_$RUNID

echo "=== SELF_CHECK biren_conformance $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"

echo '--- 0. card census (must stay free on our two cards) ---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo
echo '--- our device nodes ---'
docker exec "$C" bash -c 'ls /dev/biren/ | tr "\n" " "; echo'

echo '--- 1. sync prototype into container ---'
docker exec "$C" bash -c "rm -rf /work/dc && mkdir -p /work/dc $CB"
docker cp /tmp/proto.tgz "$C":/tmp/proto.tgz
docker exec "$C" bash -c "cd /work/dc && tar xzf /tmp/proto.tgz && echo EXTRACT_OK && ls /work/dc"

echo '--- 2. conformance 13 cases ---'
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; \
   cd /work/dc/prototype && python3 -u runtime/conformance/runner.py --backend biren --out $CB/conf13.json" \
  > /tmp/conf13.log 2>&1
RC13=$?
echo "conf13_rc=$RC13"
grep -E '^\s+\[(PASS|FAIL|ERROR|TIMEOUT)|^=== 汇总|^结论|ABORT|^\[env\]' /tmp/conf13.log | head -25

echo '--- 3. conformance infer 6 cases ---'
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; \
   cd /work/dc/prototype && python3 -u runtime/conformance/runner.py --backend biren --cases infer_cases --out $CB/conf6.json" \
  > /tmp/conf6.log 2>&1
RC6=$?
echo "conf6_rc=$RC6"
grep -E '^\s+\[(PASS|FAIL|ERROR|TIMEOUT)|^=== 汇总|^结论|ABORT|^\[env\]' /tmp/conf6.log | head -20

echo '--- 4. pull evidence back to host ---'
mkdir -p "$HOSTOUT"
docker cp "$C":"$CB"/. "$HOSTOUT"/ 2>/dev/null
cp /tmp/conf13.log /tmp/conf6.log "$HOSTOUT"/ 2>/dev/null
ls -la "$HOSTOUT"
echo "HOSTOUT=$HOSTOUT"
echo "RC13=$RC13 RC6=$RC6"
echo "=== END biren_conformance $RUNID ==="
