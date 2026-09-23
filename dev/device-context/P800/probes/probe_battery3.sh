#!/bin/bash
# 第三轮：剂量-反应——固定 KL3=1，只改「同步间隔 N（=同步前积压的通信次数）」
exec > /workspace/battery3.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
cd /workspace
PROBE=/workspace/dc_probe_rep.py
BUDGET=${BUDGET:-75}; REPS=${REPS:-120}; PORT=30700
clean() { for p in $(pgrep -f dc_probe_rep); do kill -9 $p 2>/dev/null; done; sleep 2; }
snapshot() {
  for p in $(pgrep -f dc_probe_rep); do
    awk -v pid=$p '{ s=$0; sub(/^[^)]*\) /,"",s); split(s,a," "); printf "    pid=%s utime=%s stime=%s\n", pid, a[12], a[13] }' /proc/$p/stat 2>/dev/null
  done
  for p in $(pgrep -f dc_probe_rep); do
    echo "    ---- gdb pid=$p 顶层帧 ----"
    timeout 45 gdb -p $p -batch -ex "set pagination off" -ex "bt 12" 2>&1 | grep -E "^#|flagcx|bkcl" | head -14
  done
}
run() {  # tag sync_every
  local tag="$1"; local n="$2"
  PORT=$((PORT+7))
  echo; echo "############ N=$n (每 $n 次通信后同步一次) | KL3=1 ############"
  clean
  env CUDA_VISIBLE_DEVICES=6,7 MODE=ar REPS=$REPS SYNC_EVERY=$n XPU_EVENT_KL3_ENABLE=1 \
      timeout $((BUDGET+40)) python3 -m torch.distributed.run --standalone \
      --nproc_per_node=2 --master_port=$PORT $PROBE > /workspace/r3_N${n}.log 2>&1 &
  local tpid=$!; local st=0
  for i in $(seq 1 $((BUDGET/5))); do
    sleep 5
    grep -q ALLDONE /workspace/r3_N${n}.log 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  echo "  >>> $([ $st = 1 ] && echo '✅ 完成' || echo '❌ 挂死')  完成到: $(grep -oE 'rep=[0-9]+' /workspace/r3_N${n}.log | tail -1)"
  [ $st != 1 ] && snapshot
  clean; sleep 1
}
echo "=========== 第三轮 剂量-反应 开始 $(date) ==========="
export FLAGCX_ADAPTOR=klx
for n in 1 1 2 3 5 10; do run "N${n}_$(date +%s)" $n; done
echo "=========== 第三轮结束 $(date) ==========="
