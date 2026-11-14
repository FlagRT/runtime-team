#!/usr/bin/env bash
# 验收样例 · tiering-async-ppu —— FlagMemory PPU 分层存储「异步在途轨」端到端真卡用例
# 单入口 runner（宿主侧）。**真机验证待 PTG_0 VPN 窗口**：本脚本可执行，机器
# （PTG_0 / ai-server）当前经 VPN 不可达，尚未在真机跑过——REPORT 如实标注，无结果可引。
#
# 【环境引用方式】本样例使用 case 自建运行环境目录 ./env（ENV-SPEC §3.2「各方向自建」
# 形态：thead-ppu-runtime v2 镜像 + FlagMemory@58e60a9 pip-editable 挂载）——PPU 样例
# 的验收环境与昇腾训练腿现行环境不同源，须自带装配；版本事实 = env/env.baseline.yaml。
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
#   - pytest：容器内按需 pip 安装（python3 -m pip install pytest，默认 PyPI）
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
ROOT="$(cd "$HERE/../../.." && pwd)"
ENV_DIR="$HERE/env"
CONTAINER=flagrt-fm-tiering-ppu-ptg0
PPU_CARD="${PPU_CARD:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${TIERING_CASE_LOG_DIR:-/tmp/tiering-async-ppu-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_TESTS=0; RC_SUMMARY=0; RC_ENV=0

cd "$ROOT"
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
