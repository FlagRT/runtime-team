#!/usr/bin/env bash
# 验收样例（S5 教学示范第二件）· fm-wheel-install-ppu —— 继承现行环境 + pip 安装交付组件（FM wheel）
#
# 【定位】与 fm-smoke-ascend（纯引用式）配对的教学示范：本件演示**在引用的环境之上，
#   自己额外加装交付组件**——以 FM 发布 wheel（pip 安装）装进容器再验收。消费方视角：
#   手里只有 wheel，没有 FM 源码仓。
#
# 【环境引用方式与选型教学点】不自建运行环境目录，直接引用 FM 在 PPU 的现行定档环境
#   dev/memory/cases/tiering-async-ppu/env/（flagrelease vllm0.24 定档镜像 + FlagMemory
#   源树 pip-editable 挂载，commit 守门 58e60a9）。为什么不用 thead-ppu env/v1：该镜像
#   vllm 0.19，FM connector 需 ≥0.24（RUNBOOK 已登记边界）——**引用哪个环境要看组件的
#   版本依赖**，这是做首件样例时必须核对的一步。详见 env/README.md。
#
# 【双形态对照（教学核心）】本环境默认以源树 pip-editable 提供 FM（开发者形态）；
#   本样例在其上 pip 安装发布 wheel（消费方形态）——C2 判据一前一后各核一次
#   __file__ 落点：装前=/workspace/FlagMemory（继承自环境），装后=site-packages（wheel
#   生效）。容器销毁即恢复源树形态（pip-editable 起容器重装），幂等。
#
# 【单入口】幂等。流程 = 选卡核对（ppu-smi 零负载）→ wheel sha256 自证 → up.sh --strict
#   （镜像 id + FM 挂载 commit）→ C2a 继承面（源树形态）→ docker cp + pip 装 wheel →
#   C2b 安装面（site-packages 形态）→ C3 功能面（异步搬运最小闭环，wheel 形态执行）→
#   evidence/summary.json → down + 回落核验。
#
# 【自判定】退出码 0 = 判据全过；非 0 = 有失败。末尾输出 overall: PASS/FAIL。
# 【日志】完整日志落 /tmp/fm-wheel-install-ppu-<时间戳>/；仓内只留 summary.json 与截选。
#
# 【外部依赖声明】（基线文件之外）
#   - 一张空闲 PPU 卡（PPU_CARD 必填，无缺省——每次执行前重新核对；CUDA_VISIBLE_DEVICES 授予）
#   - FM v0.3.0 wheel 已下载到宿主（WHEEL=path 传入或放 /tmp）：
#       curl -L -o /tmp/flagmemory-0.3.0-py3-none-any.whl \
#         https://github.com/FlagRT/FlagMemory/releases/download/v0.3.0/flagmemory-0.3.0-py3-none-any.whl
#     sha256 锚 = 38fa973fae7fb2984d89925e7e56daf4317aba784f360471358d64a4944f3665
#   - 机器：PTG_0（VPN 门控）；宿主 python3 <3.10 时自动 shim（本机 3.6→3.11）
#
# 用法（仓根执行，机器 = PTG_0）：
#   PPU_CARD=<n> WHEEL=/tmp/flagmemory-0.3.0-py3-none-any.whl \
#     bash dev/memory/cases/fm-wheel-install-ppu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/memory/cases/tiering-async-ppu/env"
CONTAINER=flagrt-fm-tiering-ppu-ptg0
WHEEL_SHA=38fa973fae7fb2984d89925e7e56daf4317aba784f360471358d64a4944f3665
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="/tmp/fm-wheel-install-ppu-$TS"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
FAILURES=()

die_usage() { echo "usage: PPU_CARD=<n> WHEEL=<wheel路径> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${PPU_CARD:-}" ] || die_usage
WHEEL="${WHEEL:-/tmp/flagmemory-0.3.0-py3-none-any.whl}"

echo "== fm-wheel-install-ppu $TS card=$PPU_CARD wheel=$WHEEL log=$LOG_DIR"

# 宿主默认 python3 可能 <3.10（PTG_0=3.6），dev/lib/baseline.py 需 ≥3.10：shim 前置 PATH
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

# ── 0) 前置：卡空闲 + wheel 在位与 sha256 自证 ──────────────────────────
seg="$(ppu-smi 2>/dev/null | awk -v c="$PPU_CARD" '$2==c {print; getline; print; exit}')"
echo "[INFO] ppu-smi card $PPU_CARD: $seg"
if ! echo "$seg" | grep -qE '0%'; then
  echo "[FAIL] 前置核对：卡 $PPU_CARD 非零负载，换空闲卡"; echo "overall: FAIL"; exit 2
fi
if [ ! -f "$WHEEL" ]; then
  echo "[FAIL] wheel 不存在：$WHEEL（下载方式见本文件头部外部依赖声明）"; echo "overall: FAIL"; exit 2
fi
GOT_SHA="$(sha256sum "$WHEEL" | awk '{print $1}')"
if [ "$GOT_SHA" != "$WHEEL_SHA" ]; then
  echo "[FAIL] wheel sha256 不符: got=$GOT_SHA"; echo "overall: FAIL"; exit 2
fi
echo "[PASS] wheel sha256 自证"

# ── 1) 镜像自证 ─────────────────────────────────────────────────────────
read_env() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_env ref)"; WANT_ID="$(read_env id)"
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: got=$GOT_ID"
[ "$GOT_ID" = "$WANT_ID" ] || { echo "[FAIL] 镜像 id 与基线不一致"; echo "overall: FAIL"; exit 2; }

# ── 收尾（无论成败；只收本样例所起容器）──────────────────────────────────
cleanup() {
  WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down >/dev/null 2>&1 || true
  sleep 3
  echo "[INFO] 回落核验: $(ppu-smi 2>/dev/null | awk -v c="$PPU_CARD" '$2==c {print; getline; print; exit}')"
}
trap cleanup EXIT
[ -n "$_PY_SHIM" ] && trap 'rm -rf "$_PY_SHIM"' EXIT

# ── 2) 起环境（继承：up.sh --strict，镜像 id + FM 挂载 commit 核对）──────
if ! (cd "$ROOT" && bash dev/lib/up.sh "$ENV_DIR" --strict); then
  echo "[FAIL] up.sh --strict 未通过（宿主 FlagMemory 检出须停基线 commit，见 env/README.md）"
  FAILURES+=("env-strict")
fi
docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || {
  echo "[FAIL] 环境容器未运行"; echo "overall: FAIL"; exit 1; }

fm_where() {  # $1=期望落点（source|site-packages）
  docker exec "$CONTAINER" python3 -c '
import flagmemory, sys
v = flagmemory.__version__; p = flagmemory.__file__
print("FM_VERSION", v); print("FM_PATH", p)
ok = v == "0.3.0" and (("workspace/FlagMemory" in p and "'"$1"'" == "source")
                       or ("site-packages" in p and "'"$1"'" == "site-packages"))
sys.exit(0 if ok else 1)'
}

# ── 3a) C2a 继承面：装 wheel 前，FM 由环境的源树 pip-editable 提供 ──────
# （等待以 import 成功为准——起容器 command 串行执行 pip install -e，不依赖宿主持久的
#   build.done 文件，该文件会跨容器残留、骗过等待）
for _i in $(seq 1 60); do
  docker exec "$CONTAINER" python3 -c 'import flagmemory' 2>/dev/null && break
  sleep 5
done
if fm_where source; then
  echo "[PASS] C2a 继承面: 环境源树形态生效（/workspace/FlagMemory）"
else
  echo "[FAIL] C2a 继承面"; FAILURES+=("inherit-face")
fi

# ── 3b) 加装：docker cp wheel + pip install（保留原文件名，改名 pip 拒装）──
docker cp "$WHEEL" "$CONTAINER:/tmp/flagmemory-0.3.0-py3-none-any.whl" || { echo "[FAIL] wheel 拷入容器失败"; FAILURES+=("wheel-cp"); }
if docker exec "$CONTAINER" bash -c 'python3 -m pip install --quiet /tmp/flagmemory-0.3.0-py3-none-any.whl'; then
  echo "[PASS] pip 安装完成"
else
  echo "[FAIL] pip 安装失败"; FAILURES+=("pip-install")
fi

# ── 3c) C2b 安装面：装 wheel 后，生效的确实是 wheel 那份 ────────────────
if fm_where site-packages; then
  echo "[PASS] C2b 安装面: wheel 形态生效（site-packages）"
else
  echo "[FAIL] C2b 安装面"; FAILURES+=("install-face")
fi

# ── 4) C3 功能面：分层管理真卡三用例（wheel 形态执行）────────────────────
# 测试资产取容器内挂载源树的 tests/，import 的 flagmemory 是已装 wheel——即
# 「用上游测试验收安装进来的产品代码」的消费方形态（两形态已各自实测 3/3，2026-11-19）。
# 注意：判定必须用 docker exec 的真实退出码——接 tee 管道会把退出码换成 tee 的（假 PASS 经典坑）
if docker exec -e CUDA_VISIBLE_DEVICES="$PPU_CARD" -e FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS      "$CONTAINER" bash -c 'mkdir -p /tmp/fm-smoke-evidence && cd /workspace/FlagMemory &&        python3 -m pytest -o addopts= -m realcard tests/tiering/test_ppu_tier_manager_realcard.py -q        --junitxml=/tmp/fm-smoke-evidence/junit.xml' > "$LOG_DIR/smoke.log" 2>&1; then
  tail -4 "$LOG_DIR/smoke.log"
  echo "[PASS] C3 功能面"
else
  tail -6 "$LOG_DIR/smoke.log"
  echo "[FAIL] C3 功能面"; FAILURES+=("smoke")
fi
docker cp "$CONTAINER:/tmp/fm-smoke-evidence/junit.xml" "$LOG_DIR/" >/dev/null 2>&1 || true

# ── 5) 机器可判 summary ─────────────────────────────────────────────────
python3 "$HERE/scripts/make_summary.py" "$HERE/evidence/summary.json" \
  "$GOT_ID" "$GOT_SHA" "$([ ${#FAILURES[@]} -eq 0 ] && echo pass || echo fail)" ${FAILURES[@]:-} 2>/dev/null \
  && echo "[INFO] summary 已写" || echo "[WARN] summary 写入失败（如实保留上次）"

if [ ${#FAILURES[@]} -eq 0 ]; then echo "overall: PASS"; exit 0
else echo "overall: FAIL (${FAILURES[*]})"; exit 1; fi
