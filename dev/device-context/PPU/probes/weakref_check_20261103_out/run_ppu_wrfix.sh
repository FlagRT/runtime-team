#!/bin/bash
# PPU 侧：运行时等价补丁的判据验证（P1 回调 / P2 rc / P3 功能形态 / P4 我方验收零回归）
set -u
WH=/bmcp_lvm_fs/hliu553/weakref_check_20261103   # 宿主视角（宿主机跑本脚本）
WC=/workspace/weakref_check_20261103            # 容器视角（docker exec 内）
O=$WH/out_wrfix
C=hliu553-dc-dev
mkdir -p "$O"
dex() { local tag=$1; shift; docker exec -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 "$C" \
        bash -lc "cd $WC && $*" > "$O/$tag.log" 2>&1; echo $? > "$O/$tag.rc"; }

echo "=== [0] 编译 ==="
dex build "python3 -c \"import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))\"" \
  && SUF=$(cat "$O/build.log" | tr -d '\n')
echo "  EXT_SUFFIX=$SUF"
docker exec "$C" bash -lc "cd $WC && gcc -O2 -shared -fPIC \$(python3-config --includes) -o wrfix$SUF wrfix.c" > "$O/compile.log" 2>&1
echo "  compile rc=$?"; tail -3 "$O/compile.log" | sed 's/^/    /'

echo "=== [1] P1 回调 / P3 功能形态（补丁后）==="
dex p1p3 "python3 -u wrfix_after.py"

echo "=== [2] P2 rc：Stream+weakref+被回收（补丁后，各 2 轮）==="
for r in 1 2; do dex p2_r$r "python3 -u wrfix_rc.py"; done

echo "=== 汇总 ==="
printf "%-14s %-6s %s\n" 用例 rc 关键行
for t in p1p3 p2_r1 p2_r2; do
  printf "%-14s %-6s\n" "$t" "$(cat $O/$t.rc 2>/dev/null)"
  python3 - "$O/$t.log" <<'PY'
import sys
for l in open(sys.argv[1], errors="replace").read().splitlines():
    if any(k in l for k in ("callback_fired", "wrfix", "stream_ok", "elapsed", "SANITY", "reached end")):
        print("   ", l.strip()[:160])
PY
done
