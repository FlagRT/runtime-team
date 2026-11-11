#!/bin/bash
# 910C · Arm B 单独重跑：用我们分支源码编译 FlagCX 并跑同一复现（2026-10-16）
#   背景：首轮 Arm B 无效——**构建失败**（`ATen/cuda/CUDAContext.h` 无条件包含，ascend-only 镜像无 cuda_runtime_api.h），
#         于是 Arm B 实际跑的还是镜像自带插件 ⇒ 读数与 Arm A 相同，不可用于判定。
#   本轮：构建副本加两处**构建期修复**后重编译：
#     ① makefiles/ascend.mk：移除 DAG 持久 kernel 的 3 个链接库 + -DFLAGCX_USE_ASCEND_DAG（镜像无 custom_opp）
#     ② plugin/torch/flagcx/src/backend_flagcx.cpp：`#include <ATen/cuda/CUDAContext.h>` 加 `#ifdef USE_NVIDIA_ADAPTOR` 守卫
#        （该头只被 NVIDIA 分支的 at::cuda::getCurrentCUDAStream 使用；ascend-only 镜像里编译即失败）
#   判据同 issue：退出码 0 且 `overall: PASS`。
set -u

C=${C:-dc-p9-910c-20261016}
PY=/usr/local/python3.11.15/bin/python3
HO=/mnt/raid/hliu553/dc_p9_20261016
O=/workspace/p9
SRC_O=/workspace/flagcx-src
SRC_H=/mnt/raid/hliu553/flagcx-build-20261016
TESTS=${TESTS:-20}
LOG="$HO/log/armB_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# Arm B：我们分支源码 → 编译 → 跑同一复现   $(date '+%F %T')"
echo "############################################################"
echo "  容器: $C 状态=$(docker inspect -f '{{.State.Status}}' $C 2>/dev/null)"
docker ps --format '{{.Names}}|{{.Status}}' | grep -E "^$C" || { echo "  !! 容器不在运行，先用 run_p9_c10d_910c_20261016.sh 建"; exit 2; }

echo
echo "===== [1] 构建期修复 ②：给 CUDA 头加守卫（幂等）====="
docker exec "$C" bash -lc "
set -e
F=$SRC_O/plugin/torch/flagcx/src/backend_flagcx.cpp
python3 - <<'PY'
p='$SRC_O/plugin/torch/flagcx/src/backend_flagcx.cpp'
s=open(p).read()
if '#ifdef USE_NVIDIA_ADAPTOR\n#include <ATen/cuda/CUDAContext.h>' in s:
    print('  已是守卫形态，跳过')
else:
    old='#include <ATen/cuda/CUDAContext.h>'
    new='#ifdef USE_NVIDIA_ADAPTOR\n#include <ATen/cuda/CUDAContext.h>\n#endif'
    assert s.count(old)==1, '锚点不唯一：%d' % s.count(old)
    open(p,'w').write(s.replace(old,new))
    print('  已加守卫')
PY
grep -n -A2 -B1 'ATen/cuda/CUDAContext.h' \$F | head -6
"

echo
echo "===== [2] 改动前的插件指纹（Arm A 用的那份）====="
docker exec "$C" bash -lc "
P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx
sha256sum \$P/_C*.so
echo '  未解析符号 aclrtSynchronizeEvent =' \$(nm -D --undefined-only \$P/_C*.so 2>/dev/null | grep -c aclrtSynchronizeEvent)
"

echo
echo "===== [3] 编译并安装（USE_ASCEND=1 pip install . --no-build-isolation --no-deps）====="
docker exec "$C" bash -lc "
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd $SRC_O
export USE_ASCEND=1
echo '  pip 开始' \$(date +%T)
$PY -m pip install . --no-build-isolation --no-deps 2>&1 | tail -25
echo '  pip 结束' \$(date +%T)
" > "$HO/build2.log" 2>&1
echo "  build rc=$?"
grep -aE "Successfully installed|error:|fatal error|Failed building|ERROR" "$HO/build2.log" | tail -8 | sed 's/^/    /'

echo
echo "===== [4] 改动后的插件指纹（必须与 [2] 不同）====="
docker exec "$C" bash -lc "
P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx
ls -l --time-style=+%m-%d_%H:%M \$P/_C*.so \$P/libflagcx.so
sha256sum \$P/_C*.so
echo '  未解析符号 aclrtSynchronizeEvent =' \$(nm -D --undefined-only \$P/_C*.so 2>/dev/null | grep -c aclrtSynchronizeEvent)
echo '  未解析符号 aclrtDestroyEvent     =' \$(nm -D --undefined-only \$P/_C*.so 2>/dev/null | grep -c aclrtDestroyEvent)
" 2>&1 | sed 's/^/  /'

echo
echo "===== [5] Arm B 复现（与 Arm A 完全同一条命令）====="
docker exec -w "$O" -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc \
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; cd $O && torchrun --nproc_per_node=2 flagcx_sync_test.py --iters $TESTS" \
    > "$HO/B2_ours.out" 2>&1
rc=$?
echo "  rc=$rc"
grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:" "$HO/B2_ours.out" | sed 's/^/    /'
echo
echo "===== [6] 判定 ====="
printf "  %-14s %-4s %s\n" "臂" "rc" "判定"
printf "  %-14s %-4s %s\n" "A_image" "$(cat $HO/A_image.rc 2>/dev/null | grep -oE '[0-9]+' | head -1)" "$(grep -aoE 'overall: (PASS|FAIL)' $HO/A_image.out 2>/dev/null | tail -1)"
printf "  %-14s %-4s %s\n" "B_ours" "$rc" "$(grep -aoE 'overall: (PASS|FAIL)' $HO/B2_ours.out 2>/dev/null | tail -1)"
echo "  证据: $HO/A_image.out · $HO/B2_ours.out · $HO/build2.log"
echo "===== 结束 $(date '+%F %T') ====="
