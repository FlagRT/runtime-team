#!/bin/bash
set -u
O=/bmcp_lvm_fs/hliu553/weakref_check_20261103/out
mkdir -p "$O"
C=hliu553-dc-dev
run() {
  local tag=$1 scr=$2; shift 2
  docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 "$C" \
    bash -lc "cd /workspace/weakref_check_20261103 && python3 -u $scr $*" > "$O/$tag.log" 2>&1
  echo $? > "$O/$tag.rc"
}
echo "=== PPU 缺陷态复核（v2 镜像 torch 2.10.0）==="
run cb_ctrl      ppu_cb_ctrl.py
run cb_stream    ppu_wr_callback.py
for m in wr_stream no_wr_drop wr_hold wr_tensor; do
  for r in 1 2; do run "${m}_r${r}" ppu_wr_check.py "$m"; done
done
printf "%-16s %-6s %s\n" 用例 rc 关键行
for t in cb_ctrl cb_stream wr_stream_r1 wr_stream_r2 no_wr_drop_r1 no_wr_drop_r2 wr_hold_r1 wr_hold_r2 wr_tensor_r1 wr_tensor_r2; do
  line=$(python3 - "$O/$t.log" <<'PY'
import sys
try:
    t=open(sys.argv[1],errors="replace").read()
except OSError:
    print("(no log)"); raise SystemExit
for l in t.splitlines():
    if "callback_fired" in l or "reached end" in l:
        print(l.strip()); break
PY
)
  printf "%-16s %-6s %s\n" "$t" "$(cat $O/$t.rc 2>/dev/null)" "$line"
done
