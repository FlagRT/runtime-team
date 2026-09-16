#!/bin/bash
# 第二轮：对第一轮的关键判别条件做重复验证 + 补两组对照组。
exec > /workspace/battery2.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
cd /workspace
PROBE=/workspace/dc_probe_rep.py
DEVPROBE=/workspace/dc_probe_devonly.py
BUDGET=${BUDGET:-75}
REPS=${REPS:-120}
PORT=30500

clean() { for p in $(pgrep -f "dc_probe_rep|dc_probe_devonly"); do kill -9 $p 2>/dev/null; done; sleep 2; }

snapshot() {
  echo "  --- 现场 CPU 时间 ---"
  for p in $(pgrep -f "dc_probe_rep|dc_probe_devonly"); do
    awk -v pid=$p '{ s=$0; sub(/^[^)]*\) /,"",s); split(s,a," "); printf "    pid=%s utime=%s stime=%s\n", pid, a[12], a[13] }' /proc/$p/stat 2>/dev/null
  done
  echo "  --- 现场 gdb（含模块映射，用于算偏移）---"
  local first=1
  for p in $(pgrep -f "dc_probe_rep|dc_probe_devonly"); do
    echo "    ============ gdb pid=$p ============"
    timeout 55 gdb -p $p -batch -ex "set pagination off" -ex "bt 14" -ex "info proc mappings" 2>&1 \
      | grep -vE "^(Copyright|License|GPL |There is NO|This GDB|Type \"|Reading symbols|For help|Find the GDB|For bug|Using host)" \
      | grep -E "^#|libcuda|libcudart|libbkcl|flagcx|xcudart|Thread" | head -40
    # 映射只在第一个进程取一次即可（同容器同库）
    first=0
  done
}

run_one() {  # tag cards mode [ENV=VAL ...]
  local tag="$1"; local cards="$2"; local mode="$3"; shift 3
  PORT=$((PORT+7))
  echo; echo "############ $tag | cards=$cards mode=$mode env=[$*] ############"
  clean
  env CUDA_VISIBLE_DEVICES=$cards MODE=$mode REPS=$REPS "$@" \
      timeout $((BUDGET+40)) python3 -m torch.distributed.run --standalone \
      --nproc_per_node=2 --master_port=$PORT $PROBE > /workspace/r2_${tag}.log 2>&1 &
  local tpid=$!; local st=0
  for i in $(seq 1 $((BUDGET/5))); do
    sleep 5
    grep -q ALLDONE /workspace/r2_${tag}.log 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  echo "  >>> $([ $st = 1 ] && echo '✅ 完成(未挂死)' || echo '❌ 挂死/异常')  完成到: $(grep -oE 'rep=[0-9]+' /workspace/r2_${tag}.log | tail -1)"
  [ $st != 1 ] && snapshot
  clean; sleep 1
}

run_dev() {  # tag [ENV=VAL ...]
  local tag="$1"; shift
  echo; echo "############ $tag | 单进程纯设备计算（无通信，10万次同步/1000次点）env=[$*] ############"
  clean
  env CUDA_VISIBLE_DEVICES=5 REPS=200000 "$@" timeout 100 python3 $DEVPROBE > /workspace/r2_${tag}.log 2>&1 &
  local tpid=$!; local st=0
  for i in $(seq 1 18); do
    sleep 5
    grep -q ALLDONE /workspace/r2_${tag}.log 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  echo "  >>> $([ $st = 1 ] && echo '✅ 完成' || echo '❌ 挂死/异常')"
  [ $st != 1 ] && snapshot
  clean
}

echo "=========== 第二轮开始 $(date) ==========="
export FLAGCX_ADAPTOR=klx

echo; echo "##### A 组：KL3=1（预期挂死）重复 3 次 #####"
for i in 1 2 3; do run_one "A${i}_KL3on" "6,7" ar XPU_EVENT_KL3_ENABLE=1; done

echo; echo "##### B 组：不设 KL3（预期通过）重复 3 次 #####"
for i in 1 2 3; do run_one "B${i}_KL3unset" "6,7" ar; done

echo; echo "##### C 组：显式 KL3=0（排除「未定义 vs 0」差异）#####"
run_one "C_KL3zero" "6,7" ar XPU_EVENT_KL3_ENABLE=0

echo; echo "##### D 组：KL3=1 但无通信（纯设备计算）→ 判断是否只需 KL3 #####"
run_dev "D_KL3on_devonly" XPU_EVENT_KL3_ENABLE=1

echo; echo "##### E 组：KL3=1 + 每次都同步（加重条件）#####"
run_one "E_syncEvery" "6,7" ar_sync1 XPU_EVENT_KL3_ENABLE=1

echo; echo "=========== 第二轮结束 $(date) ==========="
