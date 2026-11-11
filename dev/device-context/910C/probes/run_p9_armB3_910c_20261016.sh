#!/bin/bash
# 910C · Arm B 第三次：**清缓存后重编**（2026-10-16）
#   第二次的失败是"看起来成功"：pip 报 Successfully installed，但 `build3.log` 里**没有任何编译输出**，
#   且产出的 libflagcx.so `DT_NEEDED` 里没有 libopapi.so ⇒ **实际是复用旧构建产物（增量未重链）**，
#   sha256 与前一轮逐字相同。⇒ 判据不能只看 pip 退出码，要看**产物的链接事实**与**新时间戳**。
#   本轮：先清 build/ 与所有 .o/.so，再 `pip install --no-cache-dir`，并以 DT_NEEDED 含 libopapi.so 为判据。
set -u

C=${C:-dc-p9-910c-20261016}
PY=/usr/local/python3.11.15/bin/python3
HO=/mnt/raid/hliu553/dc_p9_20261016
O=/workspace/p9
SRC_O=/workspace/flagcx-src
SRC_H=/mnt/raid/hliu553/flagcx-build-20261016
TESTS=${TESTS:-20}
LOG="$HO/log/armB3_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Arm B（第 3 次）：清缓存重编 + 链接事实判据   $(date '+%F %T')"
echo "############################################################"

echo
echo "===== [1] 清理构建缓存（前一轮复用旧产物）====="
docker exec "$C" bash -lc "
cd $SRC_O
n_before=\$(find . -name '*.o' | wc -l)
rm -rf build plugin/torch/build ./*.egg-info plugin/torch/*.egg-info 2>/dev/null
find . -name '*.o' -delete 2>/dev/null
find . -name '*.so' -delete 2>/dev/null
echo '  清理前 .o 数='\$n_before'；清理后 .o 数='\$(find . -name '*.o' | wc -l)' .so 数='\$(find . -name '*.so' | wc -l)
"
echo "  --- 确认链接行（保留 opapi、去掉 cust_opapi）---"
grep -E "^DEVICE_LINK" "$SRC_H/makefiles/ascend.mk" | sed 's/^/    /'

echo
echo "===== [2] 重编并安装（--no-cache-dir）====="
docker exec "$C" bash -lc "
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd $SRC_O
export USE_ASCEND=1
echo '  pip 开始' \$(date +%T)
$PY -m pip install --no-cache-dir . --no-build-isolation --no-deps 2>&1 | tail -8
echo '  pip 结束' \$(date +%T)
" > "$HO/build4.log" 2>&1
echo "  pip rc=$?"
echo "  --- 编译活动自证（本轮必须**有**编译输出）---"
grep -acE "g\+\+|cc1plus|aarch64-linux-gnu-g\+\+" "$HO/build4.log" | sed 's/^/    编译命令行数=/'
grep -aE "Successfully installed|error:|fatal error" "$HO/build4.log" | tail -3 | sed 's/^/    /'

echo
echo "===== [3] 链接事实判据（关键：DT_NEEDED 必须含 libopapi.so）====="
docker exec "$C" bash -lc "
P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx
echo '  --- 安装后的 libflagcx.so ---'
ls -l --time-style=+%m-%d_%H:%M:%S \$P/libflagcx.so; sha256sum \$P/libflagcx.so
echo '  --- DT_NEEDED（ascend 相关）---'
readelf -d \$P/libflagcx.so 2>/dev/null | grep NEEDED | grep -iE 'opapi|nnop|ascendcl|hccl' | sed 's/^/    /'
echo '  --- 仍未解析的 aclnn 符号（应为 0）---'
echo '    aclnnInplaceAddGetWorkspaceSize =' \$(nm -D --undefined-only \$P/libflagcx.so 2>/dev/null | grep -c aclnnInplaceAddGetWorkspaceSize)
echo '  --- 插件 _C.so ---'; sha256sum \$P/_C*.so
echo '  --- import 自检 ---'
$PY -c \"import flagcx, os; print('    import ok:', os.path.dirname(flagcx.__file__))\" 2>&1 | tail -3
" 2>&1 | sed 's/^/  /'

echo
echo "===== [4] Arm B 复现（与 Arm A 完全同一条命令）====="
docker exec -w "$O" -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; cd $O && torchrun --nproc_per_node=2 flagcx_sync_test.py --iters $TESTS" \
    > "$HO/B4_ours.out" 2>&1
rc=$?
echo "  rc=$rc"
grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:|^=== VERDICT" "$HO/B4_ours.out" | sed 's/^/    /'
[ -s "$HO/B4_ours.out" ] || echo "    （输出为空！）"

echo
echo "===== [5] 汇总（三态：PASS/0 = 缺陷不存在；FAIL/1 = 复现；ERROR/2 = 无法下判定）====="
printf "  %-10s %-4s %s\n" "臂" "rc" "判定"
printf "  %-10s %-4s %s\n" "A_image" "$(cat $HO/A_image.rc 2>/dev/null | grep -oE '[0-9]+' | head -1)" "$(grep -aoE 'overall: (PASS|FAIL)' $HO/A_image.out 2>/dev/null | tail -1)"
v=$(grep -aoE "overall: (PASS|FAIL)" "$HO/B4_ours.out" 2>/dev/null | tail -1)
printf "  %-10s %-4s %s\n" "B_ours" "$rc" "${v:-ERROR（未出判定段）}"
echo "  证据: $HO/A_image.out · $HO/B4_ours.out · $HO/build4.log"
echo "===== 结束 $(date '+%F %T') ====="
