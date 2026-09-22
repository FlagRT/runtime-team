#!/bin/bash
# 三件事的核对：① 基线复现率 ② 不设 KL3 的「通过」是否为真（真值校验）
# ③ 「不做同步就通过」是否为假阴性（结尾仍校验真值）
exec > /workspace/verify.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
cd /workspace
PROBE=/workspace/dc_probe_verify.py
BUDGET=${BUDGET:-70}; REPS=${REPS:-120}; PORT=30900
clean() { for p in $(pgrep -f dc_probe_verify); do kill -9 $p 2>/dev/null; done; sleep 2; }

run() {  # tag cards mode [ENV=VAL ...]
  local tag="$1"; local cards="$2"; local mode="$3"; shift 3
  PORT=$((PORT+7))
  echo; echo "############ $tag | cards=$cards mode=$mode env=[$*] ############"
  clean
  env CUDA_VISIBLE_DEVICES=$cards MODE=$mode REPS=$REPS "$@" \
      timeout $((BUDGET+30)) python3 -m torch.distributed.run --standalone \
      --nproc_per_node=2 --master_port=$PORT $PROBE > /workspace/vf_${tag}.log 2>&1 &
  local tpid=$!; local st=0
  for i in $(seq 1 $((BUDGET/5))); do
    sleep 5
    grep -q ALLDONE /workspace/vf_${tag}.log 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  echo "  >>> $([ $st = 1 ] && echo '✅ 完成' || echo '❌ 挂死')  完成到: $(grep -oE 'rep=[0-9]+' /workspace/vf_${tag}.log | tail -1)"
  echo "  >>> 真值校验: $(grep -oE 'VERIFY[^\\n]*' /workspace/vf_${tag}.log | head -1)"
  if [ $st != 1 ]; then
    for p in $(pgrep -f dc_probe_verify); do
      awk -v pid=$p '{ s=$0; sub(/^[^)]*\) /,"",s); split(s,a," "); printf "      现场 pid=%s utime=%s stime=%s\n", pid, a[12], a[13] }' /proc/$p/stat 2>/dev/null
    done
  fi
  clean; sleep 1
}

echo "=========== 核对开始 $(date) ==========="
export FLAGCX_ADAPTOR=klx
echo "卡占用（用卡前必须为空闲）:"; xpu-smi 2>&1 | grep -E "MiB /" | head -16

echo; echo "##### ① 基线复现率：KL3=1 + 每10次同步 ×4 #####"
for i in 1 2 3 4; do run "A${i}_KL3on_sync10" "6,7" sync10 XPU_EVENT_KL3_ENABLE=1; done

echo; echo "##### ② 不设 KL3 的「通过」是否为真：×2（含真值校验）#####"
for i in 1 2; do run "B${i}_KL3unset_sync10" "6,7" sync10; done

echo; echo "##### ③ 「不同步就通过」是否假阴性：循环内不同步，结尾仍校验 ×2 #####"
for i in 1 2; do run "C${i}_KL3on_nosync_then_verify" "6,7" nosync XPU_EVENT_KL3_ENABLE=1; done

echo; echo "##### ④ 对照：不设 KL3 + 循环内不同步（结尾校验）×1 #####"
run "D1_KL3unset_nosync" "6,7" nosync

echo; echo "=========== 核对结束 $(date) ==========="
