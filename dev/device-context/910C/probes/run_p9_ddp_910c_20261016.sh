#!/bin/bash
# 910C · 原生 DDP 双卡自判定：镜像原版 vs 我们分支（2026-10-16，交替 A/B）
#   依据：issue「修复验收」第二条 —— 另需原生 DistributedDataParallel 双卡验证通过（原标注"待补"）。
#   切换：A = PYTHONPATH=$O/orig 影子化镜像原版；B = site-packages（我们编译版）。
set -u

C=${C:-dc-p9-910c-20261016}
PY=/usr/local/python3.11.15/bin/python3
HO=/mnt/raid/hliu553/dc_p9_20261016
O=/workspace/p9
STEPS=${STEPS:-10}
ROUNDS=${ROUNDS:-2}
LOG="$HO/log/ddp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# 原生 DDP 双卡自判定 A/B × $ROUNDS 轮   $(date '+%F %T')"
echo "############################################################"

run_one() {
    local r="$1"
    local arm="$2"
    local out="$HO/ddp_r${r}_${arm}.out"
    local pp=""
    if [ "$arm" = "A" ]; then pp="$O/orig"; fi
    docker exec -w "$O" -e ASCEND_RT_VISIBLE_DEVICES=0,1 ${pp:+-e PYTHONPATH=$pp} "$C" bash -lc \
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null; \
         cd $O && $PY -c \"import flagcx,os;print('LOADED', os.path.dirname(flagcx.__file__))\" 2>&1 | tail -1; \
         torchrun --nproc_per_node=2 ddp_native_test.py --steps $STEPS" > "$out" 2>&1
    local rc=$?
    local v
    v=$(grep -aoE "overall: (PASS|FAIL|ERROR)" "$out" | tail -1)
    local loaded
    loaded=$(grep -a "^LOADED" "$out" | tail -1 | sed 's/^LOADED //')
    printf "  round=%s arm=%s rc=%-3s %-16s loaded=%s\n" "$r" "$arm" "$rc" "${v:-无判定}" "$loaded"
    grep -aE "^  (ddp_wrapped|train_steps|loss_all_finite|ranks_param_equal|crash)" "$out" | sed 's/^/      /'
}

echo
echo "===== 交替序列 ====="
for r in $(seq 1 "$ROUNDS"); do
    run_one "$r" A
    run_one "$r" B
done

echo
echo "===== 汇总 ====="
printf "  %-6s %-22s %-22s\n" "轮次" "A 镜像原版" "B 我们分支"
for r in $(seq 1 "$ROUNDS"); do
    a=$(grep -aoE "overall: (PASS|FAIL|ERROR)" "$HO/ddp_r${r}_A.out" 2>/dev/null | tail -1)
    b=$(grep -aoE "overall: (PASS|FAIL|ERROR)" "$HO/ddp_r${r}_B.out" 2>/dev/null | tail -1)
    printf "  %-6s %-22s %-22s\n" "r$r" "${a:-无判定}" "${b:-无判定}"
done
echo
echo "===== A 臂首轮的关键报错（复现类别 ① 的原始症状）====="
grep -aE "rank0|338|_verify_params|RuntimeError|Error" "$HO/ddp_r1_A.out" 2>/dev/null | head -8 | sed 's/^/  /'
echo "===== 结束 $(date '+%F %T') ====="
