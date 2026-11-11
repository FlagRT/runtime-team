#!/usr/bin/env bash
# 壁仞 BR_0：(A) 多流探针重跑（S12 文案已按后端分支修正）+ (B) i6 预热后复测重叠率
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_stream2_$RUNID
CB=/work/out_stream_biren_$RUNID

echo "=== SELF_CHECK biren_stream_rerun $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo

docker exec "$C" bash -c "mkdir -p $CB"

echo '--- A. 多流探针重跑（rounds=5 / leak-iters=1000）---'
docker exec -e PYTHONDONTWRITEBYTECODE=1 -e DC_BACKEND=biren -e DC_TAG=_biren -e "DC_OUT_DIR=$CB" "$C" bash -c \
  ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; \
   cd /work/dc/prototype && python3 -u probes/probe_stream_semantics_full.py --rounds 5" \
  > /tmp/stream_biren2.log 2>&1
RCS=$?
echo "stream_rc=$RCS"
grep -aE '^\[env\]|^\[S[0-9]+\]|^STREAM_SEMANTICS' /tmp/stream_biren2.log | head -20

echo '--- B. i6 重叠率：连续两次调用（第二次=预热后）---'
cat > /tmp/i6w.py <<'PY'
import sys, torch
sys.path.insert(0, "/work/dc/prototype")
sys.path.insert(0, "/work/dc/prototype/runtime/conformance")
import runtime
from runtime.api.stream import Event as UE, Stream as US
import infer_cases as IC

bk = runtime.use("biren")
bk.device_count()
torch.zeros(1, device="supa")
ctx = {
    "device": "supa", "torch": torch,
    "sync": lambda: bk.synchronize(0),
    "event": lambda: UE(bk, bk.create_event()),
    "stream": lambda: US(bk, bk.create_stream()),
    "stream_ctx": lambda s: s.context(),
    "current_stream": lambda: US(bk, bk.current_stream()),
    "supports": bk.supports,
}
for tag in ("第1次（冷，含一次性编译）", "第2次（预热后）", "第3次（预热后）"):
    ok, detail = IC.case_i6_pipeline_overlap(ctx)
    print("  [%s] ok=%s | %s" % (tag, ok, detail))
PY
docker cp /tmp/i6w.py "$C":/tmp/i6w.py >/dev/null 2>&1
docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; \
   cd /work/dc/prototype && python3 -u /tmp/i6w.py" > /tmp/i6w.log 2>&1
RCI=$?
echo "i6_rc=$RCI"
grep -aE '第[123]次' /tmp/i6w.log | head -6

echo '--- pull evidence ---'
mkdir -p "$HOSTOUT"
docker cp "$C":"$CB"/. "$HOSTOUT"/ 2>/dev/null
cp /tmp/stream_biren2.log /tmp/i6w.log "$HOSTOUT"/ 2>/dev/null
ls "$HOSTOUT"
echo "HOSTOUT=$HOSTOUT RCS=$RCS RCI=$RCI"
echo "=== END biren_stream_rerun $RUNID ==="
