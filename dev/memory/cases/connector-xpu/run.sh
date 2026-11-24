#!/usr/bin/env bash
# 验收样例 · connector-xpu —— FlagMemory xpu KV connector 真卡「字节级搬运」单入口 runner（宿主侧）。
#
# 【环境引用方式】直接使用现行环境目录 dev/images/kunlun-infer-vllm/v1/env/v1
# （该目录已显式只挂 FlagMemory 只读源树到 /workspace/FlagMemory 并设
# PYTHONPATH=/workspace/FlagMemory，不挂整仓——镜像自带 vllm-plugin-FL 源树，整仓挂载
# 会遮蔽它，见该 env compose 文件头注）。
#
# 【与 connector-ascend/ppu/mlu 的差异】昆仑芯 copy backend（offload/xpu/xpu_copy_backend.py
# 的 XpuCopyBackend）走 torch.cuda 命名空间（torch_xmlir 符号重写，import torch 时自动生效，
# 不像寒武纪要显式 import torch_mlu）。
#
# 【单入口】幂等，可重复执行。流程 = 0)前置核对xpu-smi空闲卡 1)镜像自证
# 2)up.sh --strict 3)容器内跑字节级探针 4)evidence/summary.json 5)down清理
#
# 用法（仓根执行，机器 = P800-C/D）：
#   XPU_CARD=<n> bash dev/memory/cases/connector-xpu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/kunlun-infer-vllm/v1/env/v1"
CONTAINER=flagrt-kunlun-infer-v1-env-v1-p800a
AVOID_CARDS="${AVOID_CARDS:-1 2}"
IDLE_MIB_BAND=64
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CONNECTOR_CASE_LOG_DIR:-/tmp/connector-xpu-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_ENV=0; RC_PROBE=0

die_usage() { echo "usage: XPU_CARD=<n> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${XPU_CARD:-}" ] || die_usage
cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== connector-xpu $(date -Is) card=${XPU_CARD} log=$LOG_DIR"

card_line_after() { grep -A1 -E "^\|[[:space:]]+$1[[:space:]]+P800" "$2" | sed -n 2p; }
card_used_mib() { card_line_after "$1" "$2" | grep -oE "[0-9]+MiB */ *[0-9]+MiB" | head -1 | cut -d/ -f1 | tr -d ' MiB'; }
card_util() { card_line_after "$1" "$2" | grep -oE "[0-9]+%" | head -1 | tr -d ' %'; }

# ---- 0) 前置核对 ----
xpu-smi > "$LOG_DIR/xpu-smi-before.txt" 2>&1 || { echo "[FAIL] xpu-smi 不可用"; exit 2; }
for a in $AVOID_CARDS; do
  if [ "$a" = "$XPU_CARD" ]; then echo "[FAIL] 卡 $XPU_CARD 在 AVOID_CARDS（$AVOID_CARDS），换卡"; exit 2; fi
done
used="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
util="$(card_util "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
if [ -z "$used" ] || [ -z "$util" ]; then echo "[FAIL] 卡 $XPU_CARD 未出现在 xpu-smi 输出"; exit 2; fi
if [ "$used" -gt "$IDLE_MIB_BAND" ]; then echo "[FAIL] 卡 $XPU_CARD 显存 ${used}MiB 高于空载带"; exit 2; fi
if [ "$util" -ne 0 ]; then echo "[FAIL] 卡 $XPU_CARD 利用率 ${util}% 非零"; exit 2; fi
echo "[INFO] 卡 $XPU_CARD 空闲（显存 ${used}MiB，利用率 0%）"
docker ps --format '{{.Names}}' > "$LOG_DIR/docker-ps-before.txt" || true
if [ -s "$LOG_DIR/docker-ps-before.txt" ]; then
  echo "[INFO] 机器现有其他带卡容器在跑，核对不占用卡 $XPU_CARD 后继续（昆仑芯非独占限制）"
fi

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
if [ -z "$REF" ] || [ -z "$WANT_ID" ]; then echo "[FAIL] 基线 image.ref/id 读取为空"; exit 2; fi
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF"; echo "[INFO] 基线 id=$WANT_ID"; echo "[INFO] 本地 id=$GOT_ID"
if [ -z "$GOT_ID" ] || [ "$GOT_ID" != "$WANT_ID" ]; then echo "[FAIL] 镜像 id 与 env/v1 基线不一致"; exit 2; fi
echo "[PASS] 镜像 id 与 env/v1 基线一致"

# ---- 2) 起环境 ----
# 注：该env的env.baseline.yaml把FlagMemory commit钉在旧点58e60a9（connector-kunlunxin
# 建样例时的点），本次容器内装配的是现行dev-1.0@3b7cdb1——FlagMemory往前走了不是环境漂移，
# 镜像id已在步骤1单独自证一致，故不用--strict（同connector-ppu/connector-mlu先例）。
if ! XPU_CARD="$XPU_CARD" bash dev/lib/up.sh "$ENV_DIR"; then
  echo "[FAIL] up.sh 未通过"
  RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 3) 容器内跑字节级探针 ----
  PY_ENV=/root/miniconda/envs/python310_torch29_cuda/bin/python
  docker cp "$HERE/scripts/xpu_offload_byte_probe.py" "$CONTAINER:/tmp/xpu_offload_byte_probe.py"
  if docker exec "$CONTAINER" $PY_ENV /tmp/xpu_offload_byte_probe.py > "$LOG_DIR/probe.log" 2>&1; then
    RC_PROBE=0
  else
    RC_PROBE=1
  fi
  cat "$LOG_DIR/probe.log"

  # ---- 4) summary.json ----
  SUMMARY_LINE="$(grep '^SUMMARY_JSON::' "$LOG_DIR/probe.log" | tail -1 | sed 's/^SUMMARY_JSON:://')"
  if [ -n "$SUMMARY_LINE" ]; then
    python3 -c "
import json
d = json.loads('''$SUMMARY_LINE''')
d['image'] = {'ref': '$REF', 'id': '$GOT_ID'}
d['xpu_card'] = '$XPU_CARD'
json.dump(d, open('$CASE_EVIDENCE/summary.json', 'w'), ensure_ascii=False, indent=2)
print('[PASS] summary.json 已写入')
" || echo "[FAIL] summary.json 解析失败"
  else
    echo "[FAIL] 探针未产出 SUMMARY_JSON 行"
    RC_PROBE=1
  fi
fi

# ---- 5) 收尾（无论成败）----
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  XPU_CARD="$XPU_CARD" WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
sleep 5
xpu-smi > "$LOG_DIR/xpu-smi-after.txt" 2>&1 || true
used_after="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-after.txt")"
echo "[INFO] 收尾后卡 $XPU_CARD: 显存 ${used_after:-?}MiB"
echo "rc: env=$RC_ENV probe=$RC_PROBE  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_PROBE" = 0 ]; then
  echo "overall: PASS"
  exit 0
fi
echo "overall: FAIL"
[ "$RC_ENV" != 0 ] && exit "$RC_ENV"
exit "$RC_PROBE"
