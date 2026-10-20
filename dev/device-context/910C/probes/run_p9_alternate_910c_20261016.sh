#!/bin/bash
# 910C · 交替对照（A/B/A/B）：镜像原插件 vs 我们分支编译插件（2026-10-16）
#   为什么要交替：项目纪律 —— 分组跑分不清「变量效应」与「时间漂移」。
#   怎么切：把**镜像原版**的 flagcx 包解出来放 $O/orig/flagcx，用 `PYTHONPATH=$O/orig` 影子化；
#           不设 PYTHONPATH 时用 site-packages 里的**我们编译版**。
#   自证：每次运行前打印实际加载的 flagcx 路径（影子是否生效）+ _C.so 的 sha256。
set -u

C=${C:-dc-p9-910c-20261016}
IMG=${IMG:-flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64}
PY=/usr/local/python3.11.15/bin/python3
HO=/mnt/raid/hliu553/dc_p9_20261016
O=/workspace/p9
TESTS=${TESTS:-20}
ROUNDS=${ROUNDS:-2}
LOG="$HO/log/alternate_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# 交替对照 A(镜像原版) / B(我们分支) × $ROUNDS 轮   $(date '+%F %T')"
echo "############################################################"

echo
echo "===== [0] 解出镜像原版 flagcx 包到 $O/orig（供影子化）====="
docker run --rm --entrypoint bash -v "$HO:$O" "$IMG" -lc "
set -e
rm -rf $O/orig && mkdir -p $O/orig
cp -a /usr/local/python3.11.15/lib/python3.11/site-packages/flagcx $O/orig/flagcx
echo '  orig 包大小:' \$(du -sh $O/orig/flagcx | awk '{print \$1}')
sha256sum $O/orig/flagcx/_C*.so $O/orig/flagcx/libflagcx.so
" 2>&1 | tail -5

echo
echo "===== [1] 两套插件指纹对照（必须不同）====="
docker exec "$C" bash -lc "
P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx
echo '  [B 我们编译版]'; sha256sum \$P/_C*.so \$P/libflagcx.so | sed 's/^/    /'
echo '  [A 镜像原版  ]'; sha256sum $O/orig/flagcx/_C*.so $O/orig/flagcx/libflagcx.so | sed 's/^/    /'
" 2>&1 | tail -8

run_one() {  # $1=轮 $2=臂(A|B)
    # ⚠️ 必须**分行** local：`local r=$1 arm=$2 out=...${r}...` 会在同一 local 里用**外层**未绑定的 r/arm
    #    ⇒ set -u 下报 `arm: unbound variable`（同一轮 3 次 Edit 互相覆盖是另一族，不要混）。
    local r="$1"
    local arm="$2"
    local out="$HO/alt_r${r}_${arm}.out"
    local pp=""
    if [ "$arm" = "A" ]; then pp="$O/orig"; fi
    docker exec -w "$O" -e ASCEND_RT_VISIBLE_DEVICES=0,1 ${pp:+-e PYTHONPATH=$pp} "$C" bash -lc \
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; \
         cd $O && $PY -c \"import flagcx,os;print('LOADED', os.path.dirname(flagcx.__file__))\" 2>&1 | tail -1; \
         torchrun --nproc_per_node=2 flagcx_sync_test.py --iters $TESTS" > "$out" 2>&1
    local rc=$?
    local v; v=$(grep -aoE "overall: (PASS|FAIL)" "$out" | tail -1)
    local loaded; loaded=$(grep -a "^LOADED" "$out" | tail -1 | sed 's/^LOADED //')
    printf "  round=%s arm=%s rc=%-3s %-14s loaded=%s\n" "$r" "$arm" "$rc" "${v:-ERROR(NO-VERDICT)}" "${loaded}"
}

echo
echo "===== [2] 交替序列 ====="
for r in $(seq 1 "$ROUNDS"); do
    run_one "$r" A
    run_one "$r" B
done

echo
echo "===== [3] 汇总矩阵 ====="
printf "  %-8s %-24s %-24s\n" "轮次" "A 镜像原版" "B 我们分支"
for r in $(seq 1 "$ROUNDS"); do
    a=$(grep -aoE "overall: (PASS|FAIL)" "$HO/alt_r${r}_A.out" 2>/dev/null | tail -1)
    b=$(grep -aoE "overall: (PASS|FAIL)" "$HO/alt_r${r}_B.out" 2>/dev/null | tail -1)
    printf "  %-8s %-24s %-24s\n" "r$r" "${a:-ERROR（无判定段）}" "${b:-ERROR（无判定段）}"
done
echo
echo "  逐组明细（每份 .out 末尾的 VERDICT 段）---"
for f in "$HO"/alt_r*_*.out; do
    echo "  ### $(basename $f)"
    grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:" "$f" 2>/dev/null | sed 's/^/    /'
done
echo
echo "===== 结束 $(date '+%F %T') ====="
