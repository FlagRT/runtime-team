#!/bin/bash
# 910C · FlagCX c10d collective 缺陷：**镜像原版 vs 我们分支修复** 双卡对照（2026-10-16）
#   依据：dev-1.0 `dev/issues/flagcx-c10d-collectives/README.md`「修复验收」
#     · 判据：单入口命令退出码 0 且判定行 `overall: PASS`（八组全过）
#     · 三态：PASS/0 = 缺陷不存在；FAIL/1 = 缺陷复现；ERROR/2 = 无法下判定
#   用卡：die 5 + die 0（本轮逐 die 全量探测出的两张可用卡；容器内 0 基索引 ⇒ 0,1）
#   ⚠️ 不挂全部 16 张（本机纪律）：只挂自己真正要用的 die。
set -u

C=${C:-dc-p9-910c-20261016}
IMG=${IMG:-flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64}
PY=/usr/local/python3.11.15/bin/python3
HO=/mnt/raid/hliu553/dc_p9_20261016
O=/workspace/p9                                  # 容器视角（HO 挂到这里）
SRC_O=/workspace/flagcx-src                      # 我们分支源码（容器视角）
SRC_H=/mnt/raid/hliu553/flagcx-build-20261016
DIES=${DIES:-"5 0"}
TESTS=${TESTS:-20}
mkdir -p "$HO/log"
LOG="$HO/log/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# FlagCX c10d 缺陷对照：镜像原版 vs 我们分支"
echo "# 时间 $(date '+%F %T')  宿主 $(hostname)"
echo "############################################################"

echo
echo "===== [0] 前置：容器内不挂全卡、只挂 $DIES ====="
A=$(echo $DIES | awk '{print $1}'); B=$(echo $DIES | awk '{print $2}')
docker rm -f "$C" >/dev/null 2>&1 || true
docker run -dit --name "$C" \
    --network host --ipc host --shm-size 512g \
    --security-opt label=disable --cap-add CAP_SYS_PTRACE \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:rw \
    -v "$HO:$O" \
    -v "$SRC_H:$SRC_O" \
    --device "/dev/davinci$A" --device "/dev/davinci$B" \
    --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    -e HCCL_WHITELIST_DISABLE=1 \
    "$IMG" bash >/dev/null
sleep 2
echo "  容器: $C  状态=$(docker inspect -f '{{.State.Status}}' $C)  镜像=$IMG"
echo "  挂载: $HO -> $O ; $SRC_H -> $SRC_O"
echo "  挂卡: davinci$A, davinci$B（容器内索引 0,1）"

echo
echo "===== [1] 设备自证（两卡都在、能算）====="
docker exec -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc "
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
$PY -c \"
import torch, torch_npu
print('  device_count=%d' % torch.npu.device_count())
for d in range(torch.npu.device_count()):
    a = torch.randn(64,64,device='npu:%d'%d); print('  dev%d %s sum=%.1f' % (d, torch.npu.get_device_name(d), float((a@a).sum())))
\""

echo
echo "===== [2] Arm A：镜像自带 FlagCX（应复现缺陷）====="
echo "  --- 版本自证（改动前的插件 .so）---"
docker exec "$C" bash -lc "P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx; ls -l \$P/_C*.so \$P/libflagcx.so; sha256sum \$P/_C*.so"
run_arm() {  # $1=标签
    local tag=$1
    docker exec -w "$O" -e ASCEND_RT_VISIBLE_DEVICES=0,1 "$C" bash -lc \
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; \
         cd $O && torchrun --nproc_per_node=2 flagcx_sync_test.py --iters $TESTS" \
        > "$HO/${tag}.out" 2>&1
    local rc=$?
    echo "  rc=$rc" | tee -a "$HO/${tag}.rc"
    grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:|^=== VERDICT" "$HO/${tag}.out" | sed 's/^/    /'
    if grep -qa "overall: PASS" "$HO/${tag}.out"; then echo "  判定=$tag → PASS"; else
      if grep -qa "overall: FAIL" "$HO/${tag}.out"; then echo "  判定=$tag → FAIL（缺陷复现）"; else echo "  判定=$tag → ERROR（未出判定段）"; fi; fi
}
run_arm A_image

echo
echo "===== [3] 用我们分支源码编译并安装（官方命令：USE_ASCEND=1 pip install . --no-build-isolation）====="
echo "  --- 被测源码出处（构建副本自带）---"
sed 's/^/    /' "$SRC_H/SOURCE-PROVENANCE.txt"
echo "  --- 构建期唯一改动自证（DAG 链接项已移除）---"
grep -E "^DEVICE_LINK|^ADAPTOR_FLAG" "$SRC_H/makefiles/ascend.mk" | sed 's/^/    /'
echo "  源码版本自证：net.cc eventSynchronize=$(docker exec $C bash -lc "grep -c eventSynchronize $SRC_O/flagcx/core/net.cc"), backend flagcx event_->synchronize=$(docker exec $C bash -lc "grep -c 'event_->synchronize' $SRC_O/plugin/torch/flagcx/src/backend_flagcx.cpp")"
docker exec "$C" bash -lc "
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd $SRC_O
export USE_ASCEND=1
echo '  --- pip install 开始 $(date +%T) ---'
$PY -m pip install . --no-build-isolation --no-deps -q
echo '  --- pip install 结束 $(date +%T) ---'
" > "$HO/build.log" 2>&1
echo "  build rc=$?"
tail -12 "$HO/build.log" | sed 's/^/    /'

echo
echo "  --- 版本自证（改动后的插件 .so，sha256 必须与 Arm A 不同）---"
docker exec "$C" bash -lc "P=/usr/local/python3.11.15/lib/python3.11/site-packages/flagcx; ls -l \$P/_C*.so \$P/libflagcx.so; sha256sum \$P/_C*.so"

echo
echo "===== [4] Arm B：我们分支编译出的 FlagCX（应通过）====="
run_arm B_ours

echo
echo "===== [5] 汇总 ====="
printf "  %-12s %-8s %s\n" "臂" "rc" "判定"
for t in A_image B_ours; do
    rc=$(cat "$HO/${t}.rc" 2>/dev/null | tr -d 'rc= ')
    v=$(grep -aoE "overall: (PASS|FAIL)" "$HO/${t}.out" 2>/dev/null | tail -1)
    printf "  %-12s %-8s %s\n" "$t" "${rc:-?}" "${v:-（无判定段）}"
done
echo
echo "  证据：$HO/A_image.out · $HO/B_ours.out · $HO/build.log"
echo "  收尾：docker stop $C（本次**保留容器**，便于复核；确认后可删）"
echo "===== 结束 $(date '+%F %T') ====="
