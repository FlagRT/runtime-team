#!/bin/bash
# 官方 -base 镜像 · KL3 缺陷等价性对照
#
# 目的：在官方推荐镜像（harbor.baai.ac.cn/...:202608-base）上重跑现用镜像
#       （flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608）已定性的 KL3 缺陷，
#       判断该缺陷是「镜像相关」还是「厂商运行时/驱动固有」。
#
# 已定性结论（现用镜像，2026-09-14 ~ 09-20 实测）：
#   KL3=1 且存在设备侧集合通信 → 事件同步概率性永久挂死（18 次运行 16 次，≈89%）；
#   3 处自旋点均在厂商 libxpucuda.so.515.58.kunlun（偏移 +0x94080）；A/B 单变量对照：
#   不设 KL3 退出码 0、设 KL3=1 退出码 124（超时）。
#
# 本对照：同一脚本、同一用卡（XPU6,7）、同一探针与参数，唯一变量 = 镜像。
#   A 组：KL3=1 + 通信 ×3（现用镜像预期挂死）
#   B 组：不设 KL3 ×2（现用镜像预期通过，且带真值校验 2^N 精确匹配）
#
# ⚠️ 挂死进程持 GIL 自旋，**`timeout` 的 SIGTERM 无法中断**（实测存活远超预算），
#    故本脚本用「后台启动 + 轮询 ALLDONE + kill -9 强清」，不可用 timeout 前台等待。
#
# 用法（容器内）：bash /workspace/P800/probes/H_kl3_equivalence.sh
mkdir -p /workspace/out_base
exec > /workspace/out_base/kl3_ab.log 2>&1
source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda
cd /workspace
PROBE=/workspace/dc_probe_verify.py
BUDGET=${BUDGET:-100}; REPS=${REPS:-120}; PORT=${PORT:-31200}

clean() { for p in $(pgrep -f dc_probe_verify); do kill -9 $p 2>/dev/null; done; sleep 3; }

# 挂死现场：自旋进程的 CPU 时间 + 卡利用率（自旋特征：utime 大且卡 100% 但显存极小）
snapshot() {
  for p in $(pgrep -f dc_probe_verify); do
    awk -v pid=$p '{ s=$0; sub(/^[^)]*\) /,"",s); split(s,a," "); printf "      现场 pid=%s utime=%s stime=%s\n", pid, a[12], a[13] }' /proc/$p/stat 2>/dev/null
  done
  echo "      卡状态: $(xpu-smi 2>/dev/null | grep -A1 -E '^\| +[67] +P800' | grep -oE '[0-9]+MiB / 98304MiB|[0-9]+%' | tr '\n' ' ')"
}

run() {  # tag [ENV=VAL ...]
  local tag="$1"; shift
  PORT=$((PORT+7))
  echo; echo "############ $tag | cards=6,7 MODE=sync10 env=[$*] ############"
  clean
  env CUDA_VISIBLE_DEVICES=6,7 MODE=sync10 REPS=$REPS "$@" \
      timeout $((BUDGET+60)) python3 -m torch.distributed.run --standalone \
      --nproc_per_node=2 --master_port=$PORT $PROBE > /workspace/out_base/kl3_${tag}.log 2>&1 &
  local tpid=$!
  local st=0
  for i in $(seq 1 $((BUDGET/5))); do
    sleep 5
    grep -q ALLDONE /workspace/out_base/kl3_${tag}.log 2>/dev/null && { st=1; break; }
  done
  if [ $st = 1 ]; then
    echo "  >>> ✅ 完成(未挂死)"
    echo "  >>> 真值校验: $(grep -oE 'VERIFY[^,]*' /workspace/out_base/kl3_${tag}.log | head -1)"
  else
    echo "  >>> ❌ 挂死  完成到: $(grep -oE 'rep=[0-9]+' /workspace/out_base/kl3_${tag}.log | tail -1)"
    snapshot
  fi
  clean
}

echo "=========== P800 官方 -base 镜像 · KL3 等价性对照 开始 $(date) ==========="
echo "host=$(cat /etc/hostname)"
echo "用卡前各卡显存:"; xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -16

echo; echo "##### A 组：KL3=1 + 设备侧集合通信 ×3（现用镜像上 ≈89% 挂死）#####"
for i in 1 2 3; do run "A${i}_KL3on" XPU_EVENT_KL3_ENABLE=1; done

echo; echo "##### B 组：不设 KL3 ×2（现用镜像上通过且真值精确）#####"
for i in 1 2; do run "B${i}_KL3unset"; done

echo; echo "=========== 对照结束 $(date) ==========="
echo "用卡后各卡显存:"; xpu-smi 2>/dev/null | awk '/MiB \//{print}' | head -16
