#!/bin/bash
# 910C · Arm B 第二次重跑（2026-10-16）：修正构建期修复 ① 的粒度
#   第一次构建期修复砍掉了 `-lopapi -lnnopbase -lcust_opapi` 三个库 ⇒ 新 libflagcx.so 里
#   `aclnnInplaceAddGetWorkspaceSize`（P8 设备侧 reduce 用）**未解析**，import 即
#   `ImportError: undefined symbol: aclnnInplaceAddGetWorkspaceSize` ⇒ Arm B 无法下判定（ERROR）。
#   本轮实测定位（库级）：该符号**定义在 libopapi.so**（标准 CANN aclnn，镜像里有）；
#   只有 `aclnnFlagcxCollective`（DAG 持久 kernel 的自建算子）不在标准库里 ⇒ 只需砍 `-lcust_opapi`。
#   ⇒ 构建期修复 ① 修正为：**保留 `-lopapi -lnnopbase`，仅移除 `-L…/op_api/lib -lcust_opapi`**。
set -u

C=${C:-dc-p9-910c-20261016}
PY=/usr/local/python3.11.15/bin/python3
HO=/mnt/raid/hliu553/dc_p9_20261016
O=/workspace/p9
SRC_O=/workspace/flagcx-src
SRC_H=/mnt/raid/hliu553/flagcx-build-20261016
TESTS=${TESTS:-20}
LOG="$HO/log/armB2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Arm B（第 2 次）：修正链接粒度后重编 + 同一复现   $(date '+%F %T')"
echo "############################################################"

echo
echo "===== [1] 修正 makefiles/ascend.mk 的 DEVICE_LINK（幂等）====="
python3 - "$SRC_H/makefiles/ascend.mk" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
cur = "DEVICE_LINK  := -lascendcl -L$(DEVICE_HOME)/aarch64-linux/lib64"
want = "DEVICE_LINK  := -lascendcl -L$(DEVICE_HOME)/aarch64-linux/lib64 -lopapi -lnnopbase"
if want in s:
    print("  已是修正形态，跳过")
elif cur in s:
    s = s.replace(cur, want, 1)
    open(p, "w").write(s)
    print("  已修正：保留 -lopapi -lnnopbase，仅移除 -lcust_opapi")
else:
    print("  !! 锚点未命中，需人工查看"); sys.exit(3)
PY
grep -E "^DEVICE_LINK" "$SRC_H/makefiles/ascend.mk" | sed 's/^/    /'
echo "  自证：定义行含 cust_opapi = $(grep -E '^DEVICE_LINK' "$SRC_H/makefiles/ascend.mk" | grep -c cust_opapi)（应为 0）；含 opapi = $(grep -E '^DEVICE_LINK' "$SRC_H/makefiles/ascend.mk" | grep -c opapi)"

echo
echo "===== [2] 编译并安装 ====="
docker exec "$C" bash -lc "
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd $SRC_O
export USE_ASCEND=1
echo '  pip 开始' \$(date +%T)
$PY -m pip install . --no-build-isolation --no-deps 2>&1 | tail -12
echo '  pip 结束' \$(date +%T)
" > "$HO/build3.log" 2>&1
echo "  build rc=$?"
grep -aE "Successfully installed|error:|fatal error|undefined" "$HO/build3.log" | tail -6 | sed 's/^/    /'

echo
echo "===== [3] 安装后指纹 + 符号自证 ====="
docker exec "$C" bash -lc "
P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx
sha256sum \$P/_C*.so
echo '  未解析 aclnnInplaceAddGetWorkspaceSize =' \$(nm -D --undefined-only \$P/libflagcx.so 2>/dev/null | grep -c aclnnInplaceAddGetWorkspaceSize)
echo '  未解析 aclrtSynchronizeEvent（我方修复标记）=' \$(nm -D --undefined-only \$P/_C*.so 2>/dev/null | grep -c aclrtSynchronizeEvent)
echo '  未解析 aclrtDestroyEvent（我方 RAII 标记）=' \$(nm -D --undefined-only \$P/_C*.so 2>/dev/null | grep -c aclrtDestroyEvent)
echo '  import 自检:'; \$P/../../../bin/python3 -c \"import flagcx, os; print('    ok', os.path.dirname(flagcx.__file__))\" 2>&1 | tail -3
" 2>&1 | sed 's/^/  /'

echo
echo "===== [4] Arm B 复现（与 Arm A 完全同一条命令）====="
docker exec -w "$O" -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; cd $O && torchrun --nproc_per_node=2 flagcx_sync_test.py --iters $TESTS" \
    > "$HO/B3_ours.out" 2>&1
rc=$?
echo "  rc=$rc"
grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:|^=== VERDICT" "$HO/B3_ours.out" | sed 's/^/    /'
[ -s "$HO/B3_ours.out" ] || echo "    （输出为空！）"

echo
echo "===== [5] 汇总（三态口径：PASS/0 = 缺陷不存在；FAIL/1 = 复现；ERROR/2 = 无法下判定）====="
printf "  %-10s %-4s %s\n" "臂" "rc" "判定"
printf "  %-10s %-4s %s\n" "A_image" "$(cat $HO/A_image.rc 2>/dev/null | grep -oE '[0-9]+' | head -1)" "$(grep -aoE 'overall: (PASS|FAIL)' $HO/A_image.out 2>/dev/null | tail -1)"
v=$(grep -aoE "overall: (PASS|FAIL)" "$HO/B3_ours.out" 2>/dev/null | tail -1)
printf "  %-10s %-4s %s\n" "B_ours" "$rc" "${v:-ERROR（未出判定段）}"
echo "  证据: $HO/A_image.out · $HO/B3_ours.out · $HO/build3.log"
echo "===== 结束 $(date '+%F %T') ====="
