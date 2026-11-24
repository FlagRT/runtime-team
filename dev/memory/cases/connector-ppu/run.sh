#!/usr/bin/env bash
# 验收样例 · connector-ppu —— FlagMemory ppu KV connector 真卡「字节级搬运」单入口 runner（宿主侧）。
#
# 【环境引用方式】借用 dev/memory/cases/tiering-async-ppu/env（同一个 flagrelease vllm0.24
# 定档镜像 + FlagMemory pip-editable 挂载），不新建第二份会漂移的装配——与 CASE-TEMPLATE.md
# 「借用兄弟镜像目录」原则一致。
#
# 【与 connector-ascend 的差异】PPU 的 copy backend（offload/ppu/ppu_copy_backend.py 的
# DmaCopyBackend）走标准 torch.cuda 命名空间（PlatformFL 提供 CUDA 兼容层），需要先
# source /usr/local/PPU_SDK/envsetup.sh 才能用（见 scripts/run_ppu_probe.sh）。
#
# 【单入口】幂等，可重复执行。流程 =
#   0) 前置核对：ppu-smi 采集（PPU 无带卡容器独占，实际用卡由 CUDA_VISIBLE_DEVICES 授予）
#   1) 镜像自证：本地镜像 id 与 env/env.baseline.yaml 一致
#   2) up.sh <env>（镜像id已在步骤1自证；FlagMemory走容器内现行dev-1.0检出，不走该env历史pin的58e60a9，故不用--strict，避免FlagMemory commit前进被误判为环境漂移）
#   3) 等容器内 FlagMemory pip-editable 装配完成（flagmemory-build.done 标记）
#   4) 容器内跑 scripts/ppu_offload_byte_probe.py（经 scripts/run_ppu_probe.sh 完成 SDK 三件套）
#   5) 产出 evidence/summary.json
#   6) down 清理（无论成败）
#
# 用法（仓根执行，机器 = PTG_0）：
#   PPU_CARD=0 bash dev/memory/cases/connector-ppu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/memory/cases/tiering-async-ppu/env"
CONTAINER=flagrt-fm-tiering-ppu-ptg0
PPU_CARD="${PPU_CARD:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CONNECTOR_CASE_LOG_DIR:-/tmp/connector-ppu-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_ENV=0; RC_PROBE=0

cd "$ROOT"
# PTG_0 宿主 python3 可能 <3.10（dev/lib/baseline.py 需 >=3.10）：建 shim 前置 PATH
_PY_SHIM=""
if [ "$(python3 -c "import sys; print(1 if sys.version_info>=(3,10) else 0)" 2>/dev/null || echo 0)" != "1" ]; then
  for _CAND in python3.12 python3.11 python3.10; do
    if command -v "$_CAND" >/dev/null 2>&1; then
      _PY_SHIM="$(mktemp -d /tmp/flagrt-py310-shim-XXXXXX)"
      ln -s "$(command -v "$_CAND")" "$_PY_SHIM/python3"
      PATH="$_PY_SHIM:$PATH"; break
    fi
  done
fi
[ -n "$_PY_SHIM" ] && trap 'rm -rf "$_PY_SHIM"' EXIT
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== connector-ppu $(date -Is) card=$PPU_CARD log=$LOG_DIR"

# ---- 0) 前置核对 ----
if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi > "$LOG_DIR/ppu-smi-before.txt" 2>&1 || true
  echo "[INFO] ppu-smi 已采集（人工复核卡 $PPU_CARD 空闲；PPU 无带卡容器独占）"
else
  echo "[FAIL] 本机无 ppu-smi（应在 PTG_0 上执行）"; exit 2
fi

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF 基线 id=$WANT_ID 本地 id=$GOT_ID"
if [ "$GOT_ID" != "$WANT_ID" ]; then echo "[FAIL] 镜像 id 与 env 基线不一致"; exit 2; fi
echo "[PASS] 镜像 id 与 env 基线一致"

# ---- 2) 起环境（借用 tiering-async-ppu 的 env）----
if ! bash dev/lib/up.sh "$ENV_DIR"; then
  echo "[FAIL] up.sh 未通过"
  RC_ENV=2
fi

READY=0
if [ "$RC_ENV" = "0" ]; then
  for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" test -f /workspace/evidence/flagmemory-build.done 2>/dev/null; then READY=1; break; fi
    sleep 3
  done
  if [ "$READY" != "1" ]; then
    echo "[FAIL] 容器内 FlagMemory pip-editable 装配未在 180s 内完成"
    docker exec "$CONTAINER" tail -20 /workspace/evidence/flagmemory-build.log 2>/dev/null || true
    RC_PROBE=1
  fi
fi

if [ "$RC_ENV" = "0" ] && [ "$READY" = "1" ]; then
  # ---- 3) 容器内跑字节级探针 ----
  docker cp "$HERE/scripts/ppu_offload_byte_probe.py" "$CONTAINER:/tmp/ppu_offload_byte_probe.py"
  docker cp "$HERE/scripts/run_ppu_probe.sh" "$CONTAINER:/tmp/run_ppu_probe.sh"
  if docker exec -e CUDA_VISIBLE_DEVICES="$PPU_CARD" "$CONTAINER" bash /tmp/run_ppu_probe.sh > "$LOG_DIR/probe.log" 2>&1; then
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
  WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
echo "rc: env=$RC_ENV probe=$RC_PROBE  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_PROBE" = 0 ]; then
  echo "overall: PASS"
  exit 0
fi
echo "overall: FAIL"
[ "$RC_ENV" != 0 ] && exit "$RC_ENV"
exit "$RC_PROBE"
