#!/usr/bin/env bash
# 验收样例 · fm-smoke-ascend —— FlagMemory 池冒烟（昇腾 910C 真卡，3 用例）
#
# 规范：dev/images/CASE-TEMPLATE.md §0（验收样例四件套）。
#
# 【环境引用方式】不自建运行环境目录，直接引用现行环境目录
#   dev/images/ascend-train-comm/v3/env/v2（镜像 + FlagCX/flagos-device-context/FlagMemory
#   三挂载 + cann-shmem 层包）。选择理由与教学注解见本目录 env/README.md。
#
# 【单入口】幂等，可重复执行。流程 =
#   0) 前置核对：两张 die 存在且无计算进程（每张 die 输出行留痕供人工复核）
#   1) 镜像自证：docker image inspect 的 id 与 env/v2 基线 image.id 逐字一致，不一致 FAIL
#   2) up.sh <env/v2> --strict（镜像 id + 三挂载 commit + cann-shmem 装后核对）
#   3) 容器内 FlagMemory 版本核对（import flagmemory == 0.3.0）
#   4) 容器内跑池真卡冒烟 3 用例（tests/pool/test_pool_ascend_realcard.py；
#      有界预算安全契约：BYTE_LIMIT=64MiB、显式使能——判据即 FM 仓自带安全纪律）
#   5) 产出机器可判结果 evidence/summary.json（用例名→pass/fail、计数、镜像 id、FM commit）
#   6) down 清理 + die 显存回落核验（无论成败）
#
# 【自判定】退出码 0 = 三判据全过；1 = 有失败；2 = 环境不满足。末尾输出 overall: PASS/FAIL。
#
# 【日志】完整日志落 /tmp/fm-smoke-ascend-<时间戳>/（容器侧经 /tmp/fm-smoke-evidence/
#   中转，不入仓）；仓内只留 evidence/summary.json 与 REPORT 截选。
#
# 【外部依赖声明】（基线文件之外）
#   - 两张空闲 910C die（TRAIN_DIE_A / TRAIN_DIE_B 必填，无缺省——每次执行前重新核对；
#     容器内 0 号设备 = TRAIN_DIE_A）
#   - pytest：容器内按需 pip 安装（python3 -m pip install pytest，默认 PyPI）
#   - 机器：910c 系（昇腾 910C）
#
# 用法（仓根执行）：
#   TRAIN_DIE_A=0 TRAIN_DIE_B=1 bash dev/memory/cases/fm-smoke-ascend/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/ascend-train-comm/v3/env/v2"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="/tmp/fm-smoke-ascend-$TS"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
export TRAIN_DIE_A TRAIN_DIE_B
FAILURES=()

die_usage() { echo "usage: TRAIN_DIE_A=<die> TRAIN_DIE_B=<die> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${TRAIN_DIE_A:-}" ] && [ -n "${TRAIN_DIE_B:-}" ] || die_usage

fm_commit() { git -C "$ROOT/FlagMemory" rev-parse HEAD 2>/dev/null || echo unknown; }
FM_COMMIT="$(fm_commit)"

echo "== fm-smoke-ascend $TS die=${TRAIN_DIE_A}/${TRAIN_DIE_B} log=$LOG_DIR"

# ── 0) 前置核对：两张 die 存在且无计算进程 ─────────────────────────────
check_die_free() {
  local d="$1" line
  line="$(npu-smi info 2>/dev/null | awk -v d="$d" '$2==d {print; exit}')"
  if [ -z "$line" ]; then
    echo "[FAIL] 前置核对：die $d 不在 npu-smi 输出中"; return 1
  fi
  echo "[INFO] die $d: $line"
  if ! echo "$line" | grep -qE '0 */ *0($| )'; then
    echo "[FAIL] 前置核对：die $d 进程数非零（有计算进程），换空闲 die"; return 1
  fi
}
env_ok=1
check_die_free "$TRAIN_DIE_A" || env_ok=0
check_die_free "$TRAIN_DIE_B" || env_ok=0
if [ "$env_ok" != 1 ]; then echo "overall: FAIL"; exit 2; fi

# ── 1) 镜像自证（id 与 env/v2 基线逐字一致）────────────────────────────
read_env() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_env ref)"; WANT_ID="$(read_env id)"
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: got=$GOT_ID want=$WANT_ID"
if [ "$GOT_ID" != "$WANT_ID" ]; then
  echo "[FAIL] 镜像 id 与 env/v2 基线不一致（先 docker load 离线包或按配方重建）"
  echo "overall: FAIL"; exit 2
fi

# ── 收尾（无论成败都执行；只清理本样例所起容器）────────────────────────
cleanup() {
  docker compose -f "$ENV_DIR/docker-compose.yml" down >/dev/null 2>&1 || true
  sleep 3
  for d in "$TRAIN_DIE_A" "$TRAIN_DIE_B"; do
    echo "[INFO] 回落核验 die $d: $(npu-smi info 2>/dev/null | awk -v d="$d" '$2==d {print; exit}')"
  done
}
trap cleanup EXIT

# ── 2) 起环境（up.sh --strict：镜像 id + 三挂载 commit + cann-shmem 装后核对）──
if ! (cd "$ROOT" && bash dev/lib/up.sh "$ENV_DIR" --strict); then
  echo "[FAIL] up.sh --strict 未通过（宿主三仓检出须与基线 commit 一致，见 env/README.md）"
  FAILURES+=("env-strict")
fi

CONTAINER="$(docker ps --filter "name=flagrt-train-v3-env-v2-910c" -q | head -1)"
if [ -z "$CONTAINER" ]; then
  echo "[FAIL] 环境容器未运行"; echo "overall: FAIL"; exit 1
fi
echo "[INFO] container=$CONTAINER"

# ── 3) 容器内 FlagMemory 版本核对（C2）────────────────────────────────
if docker exec "$CONTAINER" python3 -c \
  'import flagmemory,sys; v=flagmemory.__version__; print("FM_VERSION",v); sys.exit(0 if v=="0.3.0" else 1)'
then echo "[PASS] C2 版本面: flagmemory==0.3.0"; else echo "[FAIL] C2 版本面"; FAILURES+=("fm-version"); fi

# ── 4) 池真卡冒烟 3 用例（C3；脚本经 stdin 注入，容器不挂仓根）──────────
echo "== 冒烟 3 用例开始 $(date -Is)"
if docker exec -i "$CONTAINER" bash -s < "$HERE/scripts/run_smoke.sh"; then
  SMOKE_RC=0; else SMOKE_RC=1; fi
docker cp "$CONTAINER:/tmp/fm-smoke-evidence" "$LOG_DIR" >/dev/null 2>&1 || true

# ── 5) 机器可判 summary（junit → summary.json）─────────────────────────
if [ "$SMOKE_RC" = 0 ]; then echo "[PASS] C3 功能面: 3/3"; else echo "[FAIL] C3 功能面"; FAILURES+=("smoke"); fi
python3 "$HERE/scripts/junit_to_summary.py" \
  "$LOG_DIR/fm-smoke-evidence/junit.xml" "$HERE/evidence/summary.json" \
  "$GOT_ID" "$FM_COMMIT" 2>/dev/null \
  && echo "[INFO] summary 已写 evidence/summary.json" \
  || { echo "[WARN] junit 缺失，summary 未更新（如实保留上次结果）"; [ "$SMOKE_RC" = 0 ] || FAILURES+=("summary"); }

# ── 判定 ────────────────────────────────────────────────────────────────
if [ ${#FAILURES[@]} -eq 0 ]; then echo "overall: PASS"; exit 0; else echo "overall: FAIL (${FAILURES[*]})"; exit 1; fi
