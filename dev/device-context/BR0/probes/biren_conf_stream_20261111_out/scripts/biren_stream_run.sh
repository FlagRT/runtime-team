#!/usr/bin/env bash
# 壁仞 BR_0：多流 Stream 16 项基线（探针覆盖的 8 项：S1/S2/S8-S13）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_stream_$RUNID
CB=/work/out_stream_biren_$RUNID

echo "=== SELF_CHECK biren_stream_semantics $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"
echo '--- card census ---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo

docker exec "$C" bash -c "mkdir -p $CB"
echo '--- probe_stream_semantics_full (backend=biren, rounds=5, leak-iters=1000) ---'
docker exec -e PYTHONDONTWRITEBYTECODE=1 -e DC_BACKEND=biren -e DC_TAG=_biren -e "DC_OUT_DIR=$CB" "$C" bash -c \
  ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; \
   cd /work/dc/prototype && python3 -u probes/probe_stream_semantics_full.py --rounds 5" \
  > /tmp/stream_biren.log 2>&1
RCS=$?
echo "stream_probe_rc=$RCS"
grep -aE '^\[env\]|^\[S[0-9]+\]|^STREAM_SEMANTICS|^  S[0-9]+_' /tmp/stream_biren.log | head -30

echo '--- pull evidence ---'
mkdir -p "$HOSTOUT"
docker cp "$C":"$CB"/. "$HOSTOUT"/ 2>/dev/null
cp /tmp/stream_biren.log "$HOSTOUT"/ 2>/dev/null
ls -la "$HOSTOUT"
echo "HOSTOUT=$HOSTOUT"
echo "RCS=$RCS"
echo "=== END biren_stream_semantics $RUNID ==="
