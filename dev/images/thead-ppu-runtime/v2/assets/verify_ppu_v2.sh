#!/bin/bash
# thead-ppu-runtime v2 —— 入档验收（证据自带 image id，杜绝"拿中间镜像的日志充数"）
#
# 用法（在 PPU 共享机宿主上跑）：
#   bash verify_ppu_v2.sh [证据输出目录]
#   默认输出目录 /bmcp_lvm_fs/hliu553/probes_ppu/v2_verify_<date>
#
# 四条判据（缺一不可；只 import 不编 kernel 不算过）：
#   ① 镜像自证：triton / flagtree 在，且与基座可区分
#   ② 非空转：真编一个 Triton kernel 上卡（TRITON_PPU_SMOKE_PASS）
#   ③ vLLM 前置门控 find_spec('flash_attn') / ('triton')
#   ④ 服务化端到端 SERVE_STANDARD_PASS（ready=1 smoke=1）
#
# 纪律：**一律用全新容器**。历史开发容器里 triton 是手工 docker cp 进去的，
#       在那种容器里验证得到的是假证据。
set -u
IMG=${IMG:-flagrt/thead-ppu-runtime:1.0.0-ppu3.6-py312-torch2.10.0-flagtree0.7.0-triton3.6.0-x86_64}
OUT=${1:-/bmcp_lvm_fs/hliu553/probes_ppu/v2_verify_$(date +%Y%m%d)}
C=hliu553-v2-verify
WS=/bmcp_lvm_fs/hliu553
P=/workspace/runtime-team/dev/device-context
mkdir -p "$OUT"
LOG=$OUT/verify.txt
: > "$LOG"

say() { echo "$@" | tee -a "$LOG"; }
run() { echo "\$ $*" >> "$LOG"; eval "$@" 2>&1 | tee -a "$LOG"; }

say "===== thead-ppu-runtime v2 入档验收 · $(date -Iseconds) ====="
say "=== [0] 被验镜像的版本锚（证据必须对应这个 id）==="
run "docker inspect $IMG --format '  image_id   = {{.Id}}'"
run "docker inspect $IMG --format '  created    = {{.Created}}'"
run "docker inspect $IMG --format '  size_bytes = {{.Size}}'"
run "docker inspect $IMG --format '  RepoDigests= {{.RepoDigests}}'"

say ""
say "=== [1] 起全新容器（独立名字，用空闲 ppu2/ppu3）==="
run "docker rm -f $C >/dev/null 2>&1 || true"
run "docker run -d --name $C --network host --restart no \
  --device=/dev/alixpu --device=/dev/alixpu_ctl \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  --shm-size=32g --add-host=ai-server:127.0.0.1 \
  -e NCCL_SOCKET_IFNAME=bond0 \
  -v $WS:/workspace -v /etc/localtime:/etc/localtime:ro \
  $IMG sleep infinity"
run "docker ps --filter name=$C --format '  {{.Names}}  {{.Status}}  {{.Image}}'"

say ""
say "=== [2] 镜像自证：triton/flagtree 在（且来自镜像层，不是手工拷贝）==="
run "docker exec $C bash -lc 'python3 - <<PY
import importlib.util as u, triton, flagtree, torch, sys
print(\"  python   :\", sys.version.split()[0])
print(\"  torch    :\", torch.__version__, \"| cuda:\", torch.cuda.is_available(), \"| n:\", torch.cuda.device_count())
print(\"  triton   :\", triton.__version__, \"| path:\", triton.__file__)
print(\"  flagtree :\", getattr(flagtree, \"__file__\", \"?\"))
print(\"  flagtree.__version__ :\", getattr(flagtree, \"__version__\", \"?\"), \"（注意：跟随 triton 线号，不作版本锚）\")
print(\"  find_spec triton/flagtree:\", u.find_spec(\"triton\") is not None, u.find_spec(\"flagtree\") is not None)
PY'"

say ""
say "=== [3] 非空转：真编 kernel（关键判据）==="
run "docker exec -e CUDA_VISIBLE_DEVICES=0 $C bash -lc 'cd $P/prototype && python3 $P/PPU/probes/triton_smoke.py'"

say ""
say "=== [4] vLLM 前置门控（基座在这里为 False ⇒ 服务化必挂）==="
run "docker exec $C bash -lc 'python3 - <<PY
import importlib.util as u
print(\"  flash_attn:\", u.find_spec(\"flash_attn\") is not None)
print(\"  triton    :\", u.find_spec(\"triton\") is not None, \" <= vLLM 门控项\")
PY'"

say ""
say "=== [5] 服务化端到端（embed 形态；STOP_AFTER=1 会在验证后主动停机，日志里的 Killed 属预期）==="
docker rm -f hliu553-v2-serve >/dev/null 2>&1 || true
SD=$OUT/serve
mkdir -p "$SD"
# ⚠️ DC_OUT_DIR 传给容器时必须是**容器视角**路径：共享盘在容器内挂在 /workspace，
#    而 $OUT 是宿主路径 /bmcp_lvm_fs/hliu553/... —— 直接传宿主路径会让脚本内的
#    重定向失败并**静默 exit 0**（2026-10-27 实测：容器 Exited(0)、日志 0 字节）。
SD_CTR=${SD/#\/bmcp_lvm_fs\/hliu553//workspace}
docker run -d --name hliu553-v2-serve --network host --restart no \
  --device=/dev/alixpu --device=/dev/alixpu_ctl \
  --device=/dev/alixpu_ppu2 --device=/dev/alixpu_ppu3 \
  --shm-size=32g --add-host=ai-server:127.0.0.1 -e NCCL_SOCKET_IFNAME=bond0 \
  -e DC_BACKEND=ppu -e DEV=0 -e STOP_AFTER=1 -e PORT=8123 -e SERVE_FORM=embed \
  -e MODEL=/workspace/models/Qwen3-Embedding-0.6B -e CUDA_VISIBLE_DEVICES=0 \
  -e DC_OUT_DIR=$SD_CTR \
  -v "$WS:/workspace" -v /etc/localtime:/etc/localtime:ro \
  "$IMG" bash -lc "cd $P/prototype && bash scripts/serve_standard.sh" >/dev/null
echo "  等待服务化判定（约 120 s）..." | tee -a "$LOG"
for i in $(seq 1 40); do
  sleep 5
  if grep -qa "SERVE_STANDARD_" "$SD/serve_standard.log" 2>/dev/null; then break; fi
done
if [ -s "$SD/serve_standard.log" ]; then
  run "grep -aE 'verdict|SERVE_STANDARD' $SD/serve_standard.log | tail -3"
else
  say "  [FAIL] $SD/serve_standard.log 缺失或为空 ⇒ 服务化判据**未取得**（不得据此判 PASS）"
fi
run "docker ps -a --filter name=hliu553-v2-serve --format '  容器状态: {{.Status}}'"

say ""
say "=== [6] 收尾：删除验证容器（不留带卡容器占别人名额）==="
run "docker rm -f $C hliu553-v2-serve >/dev/null 2>&1 || true"
run "docker ps --filter name=hliu553-v2 --format '  残留: {{.Names}}'"

say ""
say "证据目录：$OUT"
say "判据汇总："
V=$(grep -aoE "TRITON_PPU_SMOKE_PASS|SERVE_STANDARD_PASS|SERVE_STANDARD_FAIL" "$LOG" | sort -u | tr '\n' ' ')
say "  $V"
