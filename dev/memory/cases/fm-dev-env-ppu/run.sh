#!/usr/bin/env bash
# 验收样例（S5 教学示范第三件）· fm-dev-env-ppu —— 自建派生环境 + FM 开发线真卡验证
#
# 【定位】三部曲之三：①纯引用（fm-smoke-ascend）②引用+临时加装（fm-wheel-install-ppu）
#   ③**自建派生环境**（本件）——对齐 FM 在 PPU 的现行定档环境（tiering-async-ppu/env），
#   唯一差异 = FM 挂载锚切开发分支。演示「开发分支的 strict 验证」这一自建标准场景。
#
# 【顺便承载的真实特性开发】本样例的判据 = FM 分支 xliu969/fix-ppu-copyin-test 的
#   验证：该修复更正了 test_ppu_backend_realcard.py 换行换入用例的数据行
#   （2026-11-19 定位：产品行为正确、用例把数据写进了目标行而非条目记录源行，
#   PTG_0 真机 1 failed → 修复后 2 passed）。修复前后的对照即本环境的验证价值。
#
# 【单入口】幂等。流程 = 选卡核对（ppu-smi 零负载）→ 镜像自证（登记桩 id）→
#   up.sh <本样例 env> --strict（镜像 id + **FM 开发分支 commit** 守门）→
#   等待 pip-editable 就绪（import 成功为准）→ C2 锚核对（FM commit = 修复分支头）→
#   C3a 修复用例（backend 2 用例，修复前 1 failed）→ C3b 不回归（manager 3 用例）→
#   evidence/summary.json → down + 回落核验。
#
# 【自判定】退出码 0 = 判据全过；非 0 = 有失败。末尾输出 overall: PASS/FAIL。
# 【日志】完整日志落 /tmp/fm-dev-env-ppu-<时间戳>/；仓内只留 summary.json 与截选。
#
# 【外部依赖声明】（基线文件之外）
#   - 一张空闲 PPU 卡（PPU_CARD 必填；CUDA_VISIBLE_DEVICES 授予）
#   - 宿主 FlagMemory 检出停在 xliu969/fix-ppu-copyin-test（strict 守其 commit；
#     开发线推进后：ff 检出 → 重导本目录 env.baseline.yaml → 更新下方 FM_COMMIT_EXPECT）
#   - 机器：PTG_0（VPN 门控）；宿主 python3 <3.10 时自动 shim
#
# 用法（仓根执行，机器 = PTG_0）：
#   PPU_CARD=<n> bash dev/memory/cases/fm-dev-env-ppu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$HERE/env"
CONTAINER=flagrt-fm-dev-ppu-ptg0
FM_COMMIT_EXPECT="${FM_COMMIT_EXPECT:-08af9f318dc6e26cdb0796288d6fb0d1227c66ad}"   # xliu969/fix-ppu-copyin-test 线头
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="/tmp/fm-dev-env-ppu-$TS"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
FAILURES=()

die_usage() { echo "usage: PPU_CARD=<n> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${PPU_CARD:-}" ] || die_usage

echo "== fm-dev-env-ppu $TS card=$PPU_CARD fm_expect=$FM_COMMIT_EXPECT log=$LOG_DIR"

# 宿主默认 python3 可能 <3.10（PTG_0=3.6）：shim 前置 PATH
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

# ── 0) 前置：卡空闲 + 宿主 FM 检出预核 ──────────────────────────────────
seg="$(ppu-smi 2>/dev/null | awk -v c="$PPU_CARD" '$2==c {print; getline; print; exit}')"
echo "[INFO] ppu-smi card $PPU_CARD: $seg"
echo "$seg" | grep -qE '0%' || { echo "[FAIL] 前置核对：卡 $PPU_CARD 非零负载"; echo "overall: FAIL"; exit 2; }
FM_HEAD="$(git -C "$ROOT/FlagMemory" rev-parse HEAD 2>/dev/null || echo none)"
echo "[INFO] 宿主 FM 检出: $FM_HEAD（期望 $FM_COMMIT_EXPECT）"

# ── 1) 镜像自证（登记桩 id）────────────────────────────────────────────
WANT_ID="$(python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/image.anchor.yaml'))['image']['id'])")"
GOT_ID="$(docker image inspect harbor.baai.ac.cn/flagrelease-public/flagrelease_ppu_vllm024_gems_540:1012 --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: got=$GOT_ID"
[ "$GOT_ID" = "$WANT_ID" ] || { echo "[FAIL] 镜像 id 与登记桩不一致"; echo "overall: FAIL"; exit 2; }

# ── 收尾（无论成败；只收本样例容器）─────────────────────────────────────
cleanup() {
  WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down >/dev/null 2>&1 || true
  sleep 3
  echo "[INFO] 回落核验: $(ppu-smi 2>/dev/null | awk -v c="$PPU_CARD" '$2==c {print; getline; print; exit}')"
}
trap cleanup EXIT
[ -n "$_PY_SHIM" ] && trap 'rm -rf "$_PY_SHIM"' EXIT

# ── 2) 起环境（自建派生：up.sh --strict，守门 = 镜像 id + FM 开发分支 commit）──
if ! (cd "$ROOT" && bash dev/lib/up.sh "$ENV_DIR" --strict); then
  echo "[FAIL] up.sh --strict 未通过（常见原因：宿主 FM 检出不在 env.baseline 登记的 commit；开发线推进后先重导 baseline）"
  FAILURES+=("env-strict")
fi
docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || {
  echo "[FAIL] 环境容器未运行"; echo "overall: FAIL"; exit 1; }

# 等待 pip-editable 就绪（import 成功为准，不依赖 build.done 宿主残留文件）
for _i in $(seq 1 60); do
  docker exec "$CONTAINER" python3 -c 'import flagmemory' 2>/dev/null && break
  sleep 5
done

# ── 3) C2 锚核对：容器内生效的 FM = 开发分支头 ──────────────────────────
GOT_FM="$(docker exec "$CONTAINER" bash -c 'cd /workspace/FlagMemory && git rev-parse HEAD' 2>/dev/null | tail -1)"
echo "[INFO] 容器内 FM 检出: $GOT_FM"
if [ "$GOT_FM" = "$FM_COMMIT_EXPECT" ]; then
  echo "[PASS] C2 锚核对: 容器内 FM = $FM_COMMIT_EXPECT（开发分支头）"
else
  echo "[FAIL] C2 锚核对"; FAILURES+=("fm-anchor")
fi

# ── 4) C3a 修复用例（backend 2 用例；修复前 1 failed）+ C3b 不回归（manager 3 用例）──
# 注意：判定用 docker exec 真实退出码（不接 tee，防退出码被吞出假 PASS）
for SUITE in "backend:tests/tiering/test_ppu_backend_realcard.py" "manager:tests/tiering/test_ppu_tier_manager_realcard.py"; do
  NAME="${SUITE%%:*}"; FILE="${SUITE#*:}"
  if docker exec -e CUDA_VISIBLE_DEVICES="$PPU_CARD" -e FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS \
       "$CONTAINER" bash -c "cd /workspace/FlagMemory && python3 -m pytest -o addopts= -m realcard $FILE -q --junitxml=/tmp/fm-dev-$NAME-junit.xml" \
       > "$LOG_DIR/$NAME.log" 2>&1; then
    tail -2 "$LOG_DIR/$NAME.log"
    echo "[PASS] C3-$NAME"
  else
    tail -3 "$LOG_DIR/$NAME.log"
    echo "[FAIL] C3-$NAME"; FAILURES+=("suite-$NAME")
  fi
  docker cp "$CONTAINER:/tmp/fm-dev-$NAME-junit.xml" "$LOG_DIR/" >/dev/null 2>&1 || true
done

# ── 5) 机器可判 summary ─────────────────────────────────────────────────
python3 "$HERE/scripts/make_summary.py" "$HERE/evidence/summary.json" \
  "$GOT_ID" "$FM_COMMIT_EXPECT" "$([ ${#FAILURES[@]} -eq 0 ] && echo pass || echo fail)" ${FAILURES[@]:-} 2>/dev/null \
  && echo "[INFO] summary 已写" || echo "[WARN] summary 写入失败（如实保留上次）"

if [ ${#FAILURES[@]} -eq 0 ]; then echo "overall: PASS"; exit 0
else echo "overall: FAIL (${FAILURES[*]})"; exit 1; fi
