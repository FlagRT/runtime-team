#!/bin/bash
# 910C · FlagCX c10d 缺陷「修复验收」单入口脚本（2026-10-16）
#
# 用途：对**当前环境里已安装的 FlagCX** 一次性跑完 issue 要求的**两项**验收，并给出合并判定：
#   ① flagcx_sync_test.py --iters 20   → 判据：退出码 0 且 `overall: PASS`（八组全过）
#   ② ddp_native_test.py --steps 10    → 判据：退出码 0 且 `overall: PASS`（原生 DDP 双卡）
#
# 用法（**在带卡容器内**执行；容器需挂 2 张 910C 卡、且仓根已挂到 /workspace）：
#   bash dev/device-context/910C/probes/run_910c_p9_acceptance.sh
#
# 两个被测脚本按下面顺序解析，可用环境变量覆盖：
#   SYNC_TEST  ← 默认 <仓根>/dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py（镜像侧的复现脚本，不复制）
#   DDP_TEST   ← 默认 本脚本同目录/ddp_native_test.py
#
# 三态口径（CASE-TEMPLATE §2 规则 3）：
#   overall: PASS / 退出码 0 = 缺陷不存在（验收通过）
#   overall: FAIL / 退出码 1 = 缺陷复现
#   overall: ERROR / 退出码 2 = 无法下判定（环境不满足或撞到另一个问题），不计入前两者
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)          # dev/device-context/910C/probes -> 仓根
SYNC_TEST=${SYNC_TEST:-$REPO/dev/images/ascend-train-comm/v3/assets/flagcx_sync_test.py}
DDP_TEST=${DDP_TEST:-$HERE/ddp_native_test.py}
TESTS=${TESTS:-20}
STEPS=${STEPS:-10}
PY=${PY:-python3}
OUT=${OUT:-$HERE/acceptance_out}
mkdir -p "$OUT"

echo "############################################################"
echo "# FlagCX c10d 修复验收（单入口）  $(date '+%F %T')"
echo "# 仓根:     $REPO"
echo "# 复现脚本: $SYNC_TEST"
echo "# DDP 脚本: $DDP_TEST"
echo "############################################################"

for f in "$SYNC_TEST" "$DDP_TEST"; do
    if [ ! -f "$f" ]; then
        echo "  overall: ERROR —— 缺少被测脚本 $f（见本脚本头部说明；容器内须已挂仓根到 /workspace）"
        exit 2
    fi
done

echo
echo "===== [0] 自证：本次跑的是哪份 FlagCX ====="
"$PY" -c "import flagcx, os, glob, hashlib
p = os.path.dirname(flagcx.__file__)
print('  flagcx 包:', p)
for f in sorted(glob.glob(p + '/*.so')):
    print('   ', os.path.basename(f), hashlib.sha256(open(f, 'rb').read()).hexdigest()[:16])" 2>&1 | tail -6
echo "  卡: ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-（未设）}"
"$PY" -c "import torch, torch_npu; print('  device_count =', torch.npu.device_count())" 2>&1 | tail -2

echo
echo "===== [1] 验收 ①：flagcx_sync_test.py --iters $TESTS ====="
torchrun --nproc_per_node=2 "$SYNC_TEST" --iters "$TESTS" > "$OUT/sync_test.out" 2>&1
rc1=$?
grep -aE "^  [a-z_]+(\+[a-z]+)? +(PASS|FAIL)|^overall:" "$OUT/sync_test.out" | sed 's/^/    /'
v1=$(grep -aoE "overall: (PASS|FAIL)" "$OUT/sync_test.out" | tail -1 | awk '{print $2}')
echo "    rc=$rc1  判定=${v1:-ERROR（未出判定段）}"

echo
echo "===== [2] 验收 ②：ddp_native_test.py --steps $STEPS（原生 DDP 双卡）====="
torchrun --nproc_per_node=2 "$DDP_TEST" --steps "$STEPS" > "$OUT/ddp_test.out" 2>&1
rc2=$?
grep -aE "^  (ddp_wrapped|train_steps|loss_all_finite|ranks_param_equal|crashed_at)|^overall:" "$OUT/ddp_test.out" | sed 's/^/    /'
v2=$(grep -aoE "overall: (PASS|FAIL|ERROR)" "$OUT/ddp_test.out" | tail -1 | awk '{print $2}')
echo "    rc=$rc2  判定=${v2:-ERROR（未出判定段）}"

echo
echo "===== [3] 合并判定 ====="
printf "  %-14s rc=%-3s %s\n" "sync_test" "$rc1" "${v1:-ERROR}"
printf "  %-14s rc=%-3s %s\n" "ddp_native" "$rc2" "${v2:-ERROR}"
if [ "$rc1" = "0" ] && [ "$rc2" = "0" ]; then
    echo "  overall: PASS —— 两项验收均通过（退出码 0）"
    echo "  证据: $OUT/sync_test.out · $OUT/ddp_test.out"
    exit 0
elif [ "${v1:-}" = "PASS" ] || [ "${v2:-}" = "PASS" ] || [ "${v1:-}" = "FAIL" ] || [ "${v2:-}" = "FAIL" ]; then
    echo "  overall: FAIL —— 至少一项未通过（缺陷存在或修复不完整）"
    echo "  证据: $OUT/sync_test.out · $OUT/ddp_test.out"
    exit 1
else
    echo "  overall: ERROR —— 两项都未出判定段，无法下判定（环境/初始化问题）"
    echo "  证据: $OUT/sync_test.out · $OUT/ddp_test.out"
    exit 2
fi
