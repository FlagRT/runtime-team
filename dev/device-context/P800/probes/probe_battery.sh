#!/bin/bash
# 探针组：把可疑变量逐个隔离。挂死时自动抓 gdb 原生栈 + CPU 时间。
exec > /workspace/battery.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
cd /workspace
PROBE=/workspace/dc_probe_rep.py
DEVPROBE=/workspace/dc_probe_devonly.py
BUDGET=${BUDGET:-75}
REPS=${REPS:-120}
PORT=30300

clean() { for p in $(pgrep -f "dc_probe_rep|dc_probe_devonly"); do kill -9 $p 2>/dev/null; done; sleep 2; }

snapshot() {
  echo "  --- 挂死现场：CPU 时间与线程状态 ---"
  for p in $(pgrep -f "dc_probe_rep|dc_probe_devonly"); do
    awk -v pid=$p '{ s=$0; sub(/^[^)]*\) /,"",s); split(s,a," "); printf "    pid=%s state=%s utime=%s stime=%s\n", pid, a[1], a[12], a[13] }' /proc/$p/stat 2>/dev/null
    echo "    pid=$p 线程数=$(ls /proc/$p/task 2>/dev/null | wc -l)"
  done
  echo "  --- 挂死现场：gdb 原生栈（单次 attach，含所有线程）---"
  for p in $(pgrep -f "dc_probe_rep|dc_probe_devonly"); do
    echo "    ================= gdb pid=$p ================="
    timeout 50 gdb -p $p -batch -ex "set pagination off" -ex "bt 30" -ex "thread apply all bt 4" 2>&1 \
      | grep -vE "^(Copyright|License|GPL |There is NO|This GDB|Type \"|Reading symbols|For help|Find the GDB|For bug|Using host|\[New|\[Thread)" \
      | sed 's/^/      /' | head -80
    echo "    ================= end pid=$p ================="
  done
  echo "  --- 挂死现场：xpu-smi ---"
  xpu-smi 2>&1 | grep -E "MiB /" | head -16
}

run_mp() {   # name cards mode [ENV=VAL ...]
  local name="$1"; local cards="$2"; local mode="$3"; shift 3
  PORT=$((PORT+7))
  echo; echo "############ VARIANT $name | cards=$cards mode=$mode env=[$*] reps=$REPS budget=${BUDGET}s ############"
  clean
  env CUDA_VISIBLE_DEVICES=$cards MODE=$mode REPS=$REPS "$@" \
      timeout $((BUDGET+40)) python3 -m torch.distributed.run --standalone \
      --nproc_per_node=2 --master_port=$PORT $PROBE > /workspace/var_${name}.log 2>&1 &
  local tpid=$!; local st=0
  for i in $(seq 1 $((BUDGET/5))); do
    sleep 5
    grep -q ALLDONE /workspace/var_${name}.log 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  echo "  >>> 结果: $([ $st = 1 ] && echo '✅ 完成(未挂死)' || echo '❌ 未完成(疑似挂死/异常)')"
  echo "  --- 日志关键行 ---"
  grep -E "PG ready|rep=|ALLDONE|Error|error|Traceback" /workspace/var_${name}.log | tail -7
  echo "  >>> 完成到: $(grep -oE 'rep=[0-9]+' /workspace/var_${name}.log | tail -1)"
  [ $st != 1 ] && snapshot "$name"
  clean; sleep 1
}

run_dev() {
  echo; echo "############ VARIANT V8_devonly | 单进程纯设备计算 reps=2000 ############"
  clean
  env CUDA_VISIBLE_DEVICES=5 REPS=2000 timeout 110 python3 $DEVPROBE > /workspace/var_V8_devonly.log 2>&1 &
  local tpid=$!; local st=0
  for i in $(seq 1 20); do
    sleep 5
    grep -q ALLDONE /workspace/var_V8_devonly.log 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  echo "  >>> 结果: $([ $st = 1 ] && echo '✅ 完成' || echo '❌ 未完成')"
  grep -E "devonly rep=|ALLDONE|Error" /workspace/var_V8_devonly.log | tail -5
  [ $st != 1 ] && snapshot V8
  clean
}

echo "=========== 探针组开始 $(date) ==========="
echo "初始卡占用:"; xpu-smi 2>&1 | grep -E "MiB /" | head -16

export FLAGCX_ADAPTOR=klx
run_mp V1_base    "6,7" ar                  XPU_EVENT_KL3_ENABLE=1
run_mp V2_cards12 "1,2" ar                  XPU_EVENT_KL3_ENABLE=1
run_mp V3_noevent "6,7" ar
run_mp V4_gcmask  "6,7" ar                  XPU_EVENT_KL3_ENABLE=1 BKCL_GC_SIGNAL_MASK=1
run_mp V5_nosync  "6,7" ar_nosync           XPU_EVENT_KL3_ENABLE=1
run_mp V6_max     "6,7" ar_max              XPU_EVENT_KL3_ENABLE=1
run_mp V7_barrier "6,7" barrier             XPU_EVENT_KL3_ENABLE=1
run_dev

echo; echo "=========== 探针组结束 $(date) ==========="
