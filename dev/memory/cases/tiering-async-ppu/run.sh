#!/usr/bin/env bash
# 验收样例 · tiering-async-ppu —— FlagMemory PPU 分层存储「异步在途轨」端到端真卡用例
# 单入口 runner（宿主侧）。
#
# 【环境引用方式】本样例使用 case 自建运行环境目录 ./env（ENV-SPEC §3.2「各方向自建」
# 形态：flagrelease vllm0.24 定档镜像（flagrelease_ppu_vllm024_gems_540:1012，FM 仓
# flagmemory/tiering/offload/ppu/RUNBOOK.md §0 定档，暂不入 stack.lock）+
# FlagMemory@58e60a9 pip-editable 挂载）——PPU 样例的验收环境与昇腾训练腿现行环境
# 不同源，须自带装配；版本事实 = env/env.baseline.yaml（手写登记式）+
# env/image.anchor.yaml（镜像登记桩）。SDK 三件套（source envsetup + UMD/HGGC）在
# compose 与 scripts/run_tiering_ppu.sh 两级内置；thead-ppu-runtime v2 镜像内
# FM offload/ppu import 链断（vllm 0.19 无 PIN_MEMORY）为 RUNBOOK 已登记环境边界。
#
# 【单入口】幂等，可重复执行。流程 =
#   0) 前置核对：ppu-smi 挑空闲卡（PPU_CARD，缺省 0，须 0% util / ~2MiB 显存）
#   1) 镜像自证：本地镜像 id 与 env/env.baseline.yaml 的 image.id 逐字一致
#   2) up.sh ./env --strict（镜像 id＋FlagMemory commit 核对）
#   3) 容器内跑 FlagMemory tests/tiering/test_ppu_tier_manager_realcard.py 全量 3 用例
#      （口径按 FM 仓该文件头部：显式指到本文件＋FLAGMEMORY_ENABLE_LEGACY_CUDA=
#      I_ACCEPT_LEGACY_RISKS＋`-o addopts= -m realcard`；卡由 CUDA_VISIBLE_DEVICES 授予）
#   4) 产出机器可判结果 evidence/summary.json
#   5) down 清理＋零残留
#
# 【外部依赖声明】（基线文件之外）
#   - pytest：镜像内自带（7.2.0）；缺失时容器内按需 pip 安装（默认 PyPI）
#   - 一张空闲 PPU 卡（PPU_CARD，缺省 0）
#   - 机器：PTG_0 / ai-server（经 VPN 窗口可达）
#
# 【自判定】退出码 0 = 全部通过（3/3）；1 = 有失败；2 = 环境不满足。末尾输出 overall 行。
# 【日志】完整日志落 /tmp/tiering-async-ppu-<时间戳>/；仓内只留 evidence/summary.json。
#
# 用法（仓根执行，机器 = PTG_0）：
#   PPU_CARD=0 bash dev/memory/cases/tiering-async-ppu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$HERE/env"
CONTAINER=flagrt-fm-tiering-ppu-ptg0
PPU_CARD="${PPU_CARD:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${TIERING_CASE_LOG_DIR:-/tmp/tiering-async-ppu-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_TESTS=0; RC_SUMMARY=0; RC_ENV=0

cd "$ROOT"
# PTG_0 宿主默认 python3=3.6（<3.10），dev/lib/baseline.py 需 ≥3.10：建 python3→新解释器 shim 前置 PATH
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
echo "== tiering-async-ppu $(date -Is) card=$PPU_CARD log=$LOG_DIR"

# ---- 0) 前置核对：ppu-smi 空闲卡 ----
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

# ---- 2) 起环境 ----
if ! bash dev/lib/up.sh "$ENV_DIR" --strict; then
  echo "[FAIL] up.sh --strict 未通过"
  RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 2.5) 等容器内 FlagMemory pip-editable 装配完成（compose command 成功后落 done 标记）----
  READY=0
  for _ in $(seq 1 60); do
    if docker exec "$CONTAINER" test -f /workspace/evidence/flagmemory-build.done 2>/dev/null; then READY=1; break; fi
    sleep 3
  done
  if [ "$READY" != "1" ]; then
    echo "[FAIL] 容器内 FlagMemory pip-editable 装配未在 180s 内完成（或容器已退出）："
    docker exec "$CONTAINER" tail -20 /workspace/evidence/flagmemory-build.log 2>/dev/null || true
    RC_TESTS=1
  fi
fi

if [ "$RC_ENV" = "0" ] && [ "$READY" = "1" ]; then
  # ---- 3) 容器内跑 3 用例 ----
  docker exec "$CONTAINER" mkdir -p /workspace/evidence/tiering-async-ppu
  if docker exec -e FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS \
       -e CUDA_VISIBLE_DEVICES="$PPU_CARD" \
       "$CONTAINER" bash /workspace/dev/memory/cases/tiering-async-ppu/scripts/run_tiering_ppu.sh; then
    RC_TESTS=0
  else
    RC_TESTS=1
  fi
  docker cp "$CONTAINER:/workspace/evidence/tiering-async-ppu/junit.xml" "$LOG_DIR/junit.xml" >/dev/null 2>&1 || true
  docker cp "$CONTAINER:/workspace/evidence/tiering-async-ppu/pytest.log" "$LOG_DIR/pytest.log" >/dev/null 2>&1 || true

  # ---- 4) summary.json ----
  if [ -f "$LOG_DIR/junit.xml" ]; then
    if python3 "$HERE/scripts/junit_to_summary.py" "$LOG_DIR/junit.xml" \
         "$ENV_DIR/env.baseline.yaml" "$CASE_EVIDENCE/summary.json"; then
      RC_SUMMARY=0
    else
      RC_SUMMARY=1
    fi
  else
    echo "[FAIL] 未取得 junit.xml（测试未产出结果）"
    RC_SUMMARY=1
  fi
fi

# ---- 5) 收尾（无论成败）----
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[FAIL] 容器 $CONTAINER 仍存在（残留）"
else
  echo "[PASS] 容器已删净（零残留）"
fi
echo "rc: env=$RC_ENV tests=$RC_TESTS summary=$RC_SUMMARY  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_TESTS" = 0 ] && [ "$RC_SUMMARY" = 0 ]; then
  echo "overall: PASS"; exit 0
fi
echo "overall: FAIL"
if [ "$RC_ENV" != 0 ]; then exit "$RC_ENV"; fi
[ "$RC_TESTS" != 0 ] && exit "$RC_TESTS"
exit "$RC_SUMMARY"
