#!/usr/bin/env bash
# PPU 兼容性对照表 · 三级采集（只读，2026-11-05）
#
# 用法：
#   bash collect_ppu_compat_20261105.sh host      [RUNID]   # 在 PPU 宿主上跑：L0
#   bash collect_ppu_compat_20261105.sh container [RUNID]   # 在 v2 容器内跑：L1（+ 可选 L2）
#   bash collect_ppu_compat_20261105.sh all       [RUNID]   # 宿主跑 L0，再把本脚本喂给现有容器跑 L1
#
# 环境变量（可选）：
#   PPU_CONTAINER        现有 v2 容器名，默认 hliu553-dc-dev
#   DC_OUT_DIR_HOST      宿主输出目录（宿主视角），默认 /tmp/ppu_compat_out/<RUNID>/host
#   DC_OUT_DIR_CONTAINER 容器输出目录（容器视角！），默认 /tmp/ppu_compat_out/<RUNID>/container
#   PPU_WITH_DEVICE=1    容器内额外跑 L2 带卡最小算子（默认 0，避免占共享机卡）
#   PPU_PROTOTYPE_ROOT   原型根（容器内视角），仅 L2 smoke 用；未设则如实跳过
#
# 纪律：只读；按 RUNID 分目录且目录已存在时自动换名（绝不覆盖旧证据）；每条打印命令原文与 rc。
#       PYTHONDONTWRITEBYTECODE=1 ⇒ 不在仓库里留 __pycache__。
set -u
export PYTHONDONTWRITEBYTECODE=1

MODE="${1:-host}"
RUNID="${2:-ppu-compat-$(date +%Y%m%d-%H%M%S)}"
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
PPU_CONTAINER="${PPU_CONTAINER:-hliu553-dc-dev}"

if [ "$MODE" = "host" ] || [ "$MODE" = "all" ]; then
  OUT="${DC_OUT_DIR_HOST:-/tmp/ppu_compat_out/$RUNID/host}"
else
  OUT="${DC_OUT_DIR_CONTAINER:-/tmp/ppu_compat_out/$RUNID/container}"
fi

# 目录已存在 ⇒ 换名，不覆盖任何既有证据
if [ -e "$OUT" ]; then
  OUT="${OUT}-$(date +%H%M%S)"
fi
mkdir -p "$OUT" || { echo "FAIL: mkdir $OUT"; exit 2; }
[ -n "$OUT" ] || { echo "FAIL: empty OUT"; exit 2; }

log()  { printf '%s\n' "$*" | tee -a "$OUT/all.log"; }
run_sh() {
  local label="$1" cmd="$2" rc
  log "=== $label ==="
  log "\$ $cmd"
  bash -c "$cmd" >> "$OUT/all.log" 2>&1
  rc=$?
  log "[rc=$rc]"
}

log "######## PPU compat collect ########"
log "runid=$RUNID mode=$MODE out=$OUT"
log "timestamp=$(date -Is) host=$(hostname) user=$(id -un)"

collect_host() {
  log "--- L0 宿主（视角：HOST，$OUT）---"
  run_sh "L0-1 宿主 os-release（P0 唯一缺口）" 'grep -E "^(PRETTY_NAME|NAME|VERSION|ID|VERSION_ID)=" /etc/os-release'
  run_sh "L0-2 宿主内核" 'uname -a'
  run_sh "L0-3 厂商工具首段（ppu-smi）" 'ppu-smi 2>&1 | head -30'
  run_sh "L0-4 厂商工具首段（nvidia-smi 符号链接）" 'nvidia-smi 2>&1 | head -30'
  run_sh "L0-5 驱动模块 / 设备节点" 'ls -l /dev/alixpu* 2>&1 | head -25'
  run_sh "L0-6 共享盘水位（11-03 曾 100% 满）" 'df -h /bmcp_lvm_fs / /data 2>&1'
  run_sh "L0-7 容器清单（确认哪个是自己的）" 'docker ps -a --format "{{.Names}}|{{.Image}}|{{.Status}}" 2>&1 | head -20'
  run_sh "L0-8 v2 镜像 id（应与 image.baseline L12 对齐）" 'docker images --digests 2>&1 | grep -E "thead-ppu-runtime|REPOSITORY" | head -10'
}

collect_container() {
  log "--- L1 容器（视角：CONTAINER，$OUT）---"
  run_sh "L1-1 容器 os-release" 'grep -E "^(PRETTY_NAME|NAME|VERSION|ID|VERSION_ID)=" /etc/os-release'
  run_sh "L1-2 容器内看到的宿主内核" 'uname -r'
  run_sh "L1-3 python / torch" 'python3 -c "import sys,torch; print(sys.version); print(torch.__version__); print(torch.version.cuda)"'
  run_sh "L1-4 包版本（torch/triton/flag/vllm/transformers）" 'pip list 2>/dev/null | grep -Ei "^(torch|triton|flag|vllm|transformers|flagtree)"'
  run_sh "L1-5 triton / flag_gems 可导入性（v2 修法核心）" 'python3 -c "import importlib.util as u; print(\"triton\", u.find_spec(\"triton\")); print(\"flag_gems\", u.find_spec(\"flag_gems\")); import triton; print(\"triton.__version__\", getattr(triton,\"__version__\",\"NA\"))"'
  run_sh "L1-6 SDK / 通信库登记路径" 'ls -ld /usr/local/PPU_SDK /usr/local/PPU_SDK/CUDA_SDK /opt/accl-p 2>&1; ls -l /usr/lib/x86_64-linux-gnu/libnccl-net* 2>&1'
  run_sh "L1-7 SDK 版本文件（如存在）" 'find /usr/local/PPU_SDK -maxdepth 3 -iname "*version*" -o -iname "*release*" 2>/dev/null | head -20'
  run_sh "L1-8 选卡变量与设备可见性（不带卡，仅回读）" 'printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES-<unset>}"; printf "NCCL_SOCKET_IFNAME=%s\n" "${NCCL_SOCKET_IFNAME-<unset>}"'
  run_sh "L1-9 dispatch 注册来源（对照 910C §3.1e）" 'python3 -c "import torch; print(\"CUDA registered:\", torch._C._dispatch_dump_table(\"CUDA\")[:1500] if hasattr(torch._C,\"_dispatch_dump_table\") else \"no dump_table\")"'

  if [ "${PPU_WITH_DEVICE:-0}" = "1" ]; then
    log "--- L2 带卡（显式开启，默认关闭）---"
    run_sh "L2-1 device_count + 最小算子" 'python3 -c "import torch; n=torch.cuda.device_count(); print(\"device_count\",n); x=torch.zeros(64,64,device=\"cuda\"); y=(x+1).matmul(x+1); torch.cuda.synchronize(); print(\"minop_ok\", float(y.sum().item()))"'
    if [ -n "${PPU_PROTOTYPE_ROOT:-}" ] && [ -f "${PPU_PROTOTYPE_ROOT}/runtime/smoke_runtime.py" ]; then
      run_sh "L2-2 本层 smoke" "cd ${PPU_PROTOTYPE_ROOT} && DC_ROOT=${PPU_PROTOTYPE_ROOT} python3 runtime/smoke_runtime.py --backend ppu"
    else
      log "[skip] L2-2 smoke：PPU_PROTOTYPE_ROOT 未设或缺少 runtime/smoke_runtime.py（如实跳过，不并入结论）"
    fi
  else
    log "[skip] L2 带卡探针：PPU_WITH_DEVICE 未设 1（共享机，默认不占卡）"
  fi
}

case "$MODE" in
  host)     collect_host ;;
  container) collect_container ;;
  all)
    collect_host
    if ! docker inspect -f '{{.State.Running}}' "$PPU_CONTAINER" 2>/dev/null | grep -q true; then
      log "[skip] 容器 $PPU_CONTAINER 未运行 ⇒ 未自动 start（共享机不擅自起容器），请手动启动后重跑 container 模式"
      exit 0
    fi
    log "--- 把同一份脚本喂给容器 $PPU_CONTAINER（视角分离：宿主 $OUT / 容器 /tmp）---"
    docker exec -i \
      -e RUNID="$RUNID" \
      -e DC_OUT_DIR_CONTAINER="/tmp/ppu_compat_out/$RUNID/container" \
      -e PPU_WITH_DEVICE="${PPU_WITH_DEVICE:-0}" \
      -e PPU_PROTOTYPE_ROOT="${PPU_PROTOTYPE_ROOT:-}" \
      "$PPU_CONTAINER" bash -s container "$RUNID" < "$SELF" \
      >> "$OUT/docker_exec_container.log" 2>&1
    rc=$?
    log "[docker exec rc=$rc] 容器内原始输出：宿主 $OUT/docker_exec_container.log（容器自身输出在 /tmp/ppu_compat_out/$RUNID/container/all.log，需 docker cp 取回）"
    ;;
  *) echo "用法: $0 host|container|all [RUNID]"; exit 2 ;;
esac

log "######## DONE ########"
log "证据目录：$OUT （只追加，未覆盖任何既有文件）"
