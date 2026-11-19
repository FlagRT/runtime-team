#!/usr/bin/env bash
# 验收样例（S5 教学示范第二件）· fm-wheel-install-ppu —— 继承现行环境 + pip 安装交付组件（FM wheel）
#
# 【定位】与 fm-smoke-ascend（纯引用式）配对的教学示范：本件演示**在引用的现行环境之上，
#   自己额外加装交付组件**——环境不挂 FlagMemory（thead env/v1 的 mounted_repos 为空），
#   以发布 wheel（pip 安装）形态把它装进容器再验收。消费方视角：手里只有 wheel，没有 FM 源码仓。
#
# 【环境引用方式】不自建运行环境目录，直接引用现行环境目录
#   dev/images/thead-ppu-runtime/v2/env/v1（派生镜像 + 全 16 卡映射 + network host；
#   该环境 mounted_repos 为空 ⇒ 干净基底，正好承载"继承之上加装"的演示）。见 env/README.md。
#
# 【加装方式】docker cp wheel 进容器 → pip install（容器 network 形态不限；wheel 由宿主提供）。
#   对照：环境级登记加装请用 compose 的 x-flagrt.env_packages（层包机制， cann-shmem 先例）；
#   源码开发形态请用 mounted_repos 的 pip-editable。三种方式的取舍见 env/README.md 教学注解。
#
# 【单入口】幂等。流程 = 选卡核对（ppu-smi 零负载零显存）→ wheel 在位与 sha256 自证 →
#   镜像自证 → up.sh --strict → pip 安装（docker cp + pip install）→ C2 安装面
#   （版本==0.3.0 且模块路径在 site-packages，证明 wheel 形态生效）→ C3 功能面
#   （异步搬运最小闭环，脚本经 stdin 注入）→ evidence/summary.json → down + 回落核验。
#
# 【自判定】退出码 0 = 三判据全过；非 0 = 有失败。末尾输出 overall: PASS/FAIL。
# 【日志】完整日志落 /tmp/fm-wheel-install-ppu-<时间戳>/；仓内只留 summary.json 与截选。
#
# 【外部依赖声明】（基线文件之外）
#   - 一张空闲 PPU 卡（PPU_CARD 必填，无缺省——每次执行前重新核对；CUDA_VISIBLE_DEVICES 授予）
#   - FM v0.3.0 wheel 已下载到宿主（WHEEL=path 传入或放样例目录外 /tmp）：
#       curl -L -o /tmp/flagmemory-0.3.0-py3-none-any.whl \
#         https://github.com/FlagRT/FlagMemory/releases/download/v0.3.0/flagmemory-0.3.0-py3-none-any.whl
#     sha256 锚 = 38fa973fae7fb2984d89925e7e56daf4317aba784f360471358d64a4944f3665
#   - 机器：PTG_0（VPN 门控；thead-ppu-runtime v2 镜像已预载）
#
# 用法（仓根执行，机器 = PTG_0）：
#   PPU_CARD=<n> WHEEL=/tmp/flagmemory-0.3.0-py3-none-any.whl \
#     bash dev/memory/cases/fm-wheel-install-ppu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/thead-ppu-runtime/v2/env/v1"
CONTAINER=flagrt-ppu-v2-env-v1-ptg0
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

# ── 0) 前置：卡空闲 + wheel 在位与 sha256 自证 ──────────────────────────
line="$(ppu-smi 2>/dev/null | sed -n "$((PPU_CARD + 1))p")"
echo "[INFO] ppu-smi card $PPU_CARD: $line"
if ! echo "$line" | grep -qE '0%'; then
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
  echo "[INFO] 回落核验: $(ppu-smi 2>/dev/null | sed -n "$((PPU_CARD + 1))p")"
}
trap cleanup EXIT

# ── 2) 起环境（继承：up.sh --strict）────────────────────────────────────
if ! (cd "$ROOT" && bash dev/lib/up.sh "$ENV_DIR" --strict); then
  echo "[FAIL] up.sh --strict 未通过"; FAILURES+=("env-strict")
fi
docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || {
  echo "[FAIL] 环境容器未运行"; echo "overall: FAIL"; exit 1; }

# ── 3) 加装：docker cp wheel + pip install（教学核心步骤）───────────────
docker cp "$WHEEL" "$CONTAINER:/tmp/fm.whl" || { echo "[FAIL] wheel 拷入容器失败"; FAILURES+=("wheel-cp"); }
if docker exec "$CONTAINER" bash -c 'python3 -m pip install --quiet /tmp/fm.whl'; then
  echo "[PASS] pip 安装完成"
else
  echo "[FAIL] pip 安装失败"; FAILURES+=("pip-install")
fi

# ── 4) C2 安装面：版本 + 模块路径在 site-packages（证明 wheel 形态）──────
if docker exec "$CONTAINER" python3 -c '
import flagmemory, sys
v = flagmemory.__version__; p = flagmemory.__file__
print("FM_VERSION", v); print("FM_PATH", p)
sys.exit(0 if v == "0.3.0" and "site-packages" in p else 1)'; then
  echo "[PASS] C2 安装面: wheel 形态生效（site-packages）"
else
  echo "[FAIL] C2 安装面"; FAILURES+=("install-face")
fi

# ── 5) C3 功能面：异步搬运最小闭环（PPU SDK 环境 + 选卡授予）────────────
if docker exec -e CUDA_VISIBLE_DEVICES="$PPU_CARD" -i "$CONTAINER" bash -c \
  'source /usr/local/PPU_SDK/envsetup.sh && export UMD_PLATFORM_TYPE=1 HGGC_DRIVER_CANDIDATE=UMD && python3 -'; \
  < "$HERE/scripts/smoke_e2e.py" | tee "$LOG_DIR/smoke.log"; then
  echo "[PASS] C3 功能面"
else
  echo "[FAIL] C3 功能面"; FAILURES+=("smoke")
fi

# ── 6) 机器可判 summary ─────────────────────────────────────────────────
C1=$([ ${#FAILURES[@]} -eq 0 ] || [[ " ${FAILURES[*]} " != *"env-strict"* ]] && echo pass || echo fail)
python3 "$HERE/scripts/make_summary.py" "$HERE/evidence/summary.json" \
  "$GOT_ID" "$GOT_SHA" "$C1" "${FAILURES[@]:-}" 2>/dev/null \
  && echo "[INFO] summary 已写" || echo "[WARN] summary 写入失败（如实保留上次）"

if [ ${#FAILURES[@]} -eq 0 ]; then echo "overall: PASS"; exit 0
else echo "overall: FAIL (${FAILURES[*]})"; exit 1; fi
