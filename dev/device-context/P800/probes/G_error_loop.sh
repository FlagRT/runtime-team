#!/bin/bash
# P800 阶段 4 · 错误注入 → 恢复闭环（设备侧三段：识别 / 处置 / 业务继续）
#
# 关键设计：**两设置对照** —— 同一脚本、同一用卡，唯一变量 XPU_EVENT_KL3_ENABLE（设 / 不设）。
#   理由：该变量是厂商事件同步开关，可能影响设备异常上报路径，而错误捕获属我方五域职责
#         ⇒ 「关掉它是否损失诊断能力」这一差异本身就是产出。
#   注意：错误闭环为**单进程单卡**、不走集合通信；KL3 缺陷的两要素是「KL3=1 + 设备侧集合通信」，
#         故本对照预期两组都能跑完；若 KL3=1 组挂死，则是对该缺陷边界的重要补充证据。
#
# 用法（容器内）：  DEV=6 bash G_error_loop.sh
set -u
mkdir -p /workspace/out_err
exec > /workspace/out_err/G_error_loop.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda

DEV=${DEV:-6}
BUDGET=${BUDGET:-600}
PROTO=/workspace/prototype/runtime/proto/proto_error_recovery_loop.py
OUT=/workspace/out_err

clean() { for p in $(pgrep -f proto_error_recovery_loop); do kill -9 $p 2>/dev/null; done; sleep 2; }

snapshot() {
  echo "    ---- 挂死进程状态（判断是自旋还是阻塞）----"
  for p in $(pgrep -f proto_error_recovery_loop); do
    awk -v pid=$p '{ s=$0; sub(/^[^)]*\) /,"",s); split(s,a," "); printf "    pid=%s utime=%s stime=%s\n", pid, a[12], a[13] }' /proc/$p/stat 2>/dev/null
  done
  for p in $(pgrep -f proto_error_recovery_loop); do
    echo "    ---- gdb pid=$p 顶层帧 ----"
    timeout 45 gdb -p $p -batch -ex "set pagination off" -ex "bt 10" 2>&1 \
      | grep -E "^#|flagcx|bkcl|xpucuda" | head -12
  done
}

run() {  # tag  kl3_setting(0/1)
  local tag="$1"; local kl3="$2"
  local log=$OUT/G_${tag}.log
  echo; echo "############ 错误闭环 · KL3 ${kl3} · $tag ############"
  clean
  if [ "$kl3" = "1" ]; then
    env CUDA_VISIBLE_DEVICES=$DEV DC_BACKEND=kunlun XPU_EVENT_KL3_ENABLE=1 \
        FLAGCX_ADAPTOR=klx timeout $BUDGET python3 $PROTO --backend kunlun \
        --out $OUT/error_recovery_loop_kunlun_KL3on.json > "$log" 2>&1 &
  else
    env -u XPU_EVENT_KL3_ENABLE CUDA_VISIBLE_DEVICES=$DEV DC_BACKEND=kunlun \
        FLAGCX_ADAPTOR=klx timeout $BUDGET python3 $PROTO --backend kunlun \
        --out $OUT/error_recovery_loop_kunlun_KL3off.json > "$log" 2>&1 &
  fi
  local tpid=$!; local st=0
  for i in $(seq 1 $((BUDGET/5))); do
    sleep 5
    grep -q "ERROR_RECOVERY_LOOP" "$log" 2>/dev/null && { st=1; break; }
    kill -0 $tpid 2>/dev/null || { st=2; break; }
  done
  if [ $st = 1 ]; then
    echo "  >>> ✅ 完成：$(grep -oE '闭环 [0-9]+ / 跳过 [0-9]+ / 失败 [0-9]+' "$log" | tail -1)"
    grep -E "^\[" "$log" | tail -8 | sed 's/^/     /'
  else
    echo "  >>> ❌ 挂死/异常退出（rc 见日志）"
    snapshot
  fi
  clean; sleep 1
}

echo "=========== P800 阶段 4 错误闭环（两设置对照）开始 $(date) ==========="
echo "用卡: $DEV | 超时: ${BUDGET}s | 唯一变量: XPU_EVENT_KL3_ENABLE"
echo "--- 用卡前：各卡显存占用 ---"
xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8

export FLAGCX_ADAPTOR=klx
run noKL3 0
run KL3on 1

echo
echo "--- 对照组差异（原始结果）---"
for f in $OUT/error_recovery_loop_kunlun_KL3off.json $OUT/error_recovery_loop_kunlun_KL3on.json; do
  [ -f "$f" ] && echo "  $f: $(python3 -c "
import json,sys
d=json.load(open('$f'))
print('closed', d['closed_loops'], 'skipped', d['skipped'], 'failed', d['failed'], '|', d['verdict'])
" 2>&1 | tail -1)"
done
echo "--- 用卡后复查 ---"
xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -8
clean
echo "=========== 阶段 4 错误闭环 结束 $(date) ==========="
