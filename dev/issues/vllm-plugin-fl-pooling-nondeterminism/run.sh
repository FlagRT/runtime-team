#!/usr/bin/env bash
# vllm-plugin-fl-pooling-nondeterminism —— 单入口 runner（宿主侧，收数脚本，不判定）。
#
# 环境与模型依赖同 [`../vllm-plugin-fl-pooling-dispatch/run.sh`](../vllm-plugin-fl-pooling-dispatch/run.sh)
# （同一现行环境目录 dev/images/kunlun-infer-vllm/v1/env/v1，同一镜像、同一容器）。
#
# 【这是收数脚本，不是判定脚本】根因未知，没有验收判据——流程 = 起环境 → 跑两组对照
# （默认 enable_prefix_caching=True / 显式 False）→ 把 cos 值写进 evidence/summary.json →
# 清理。退出码 0 = 收数完成，2 = 执行出错；**都不代表任何修复验收结论**。
#
# 【外部依赖声明】同上一个 case：XPU_CARD、MODEL_DIR 必填，无缺省。
#
# 用法（仓根执行）：
#   XPU_CARD=<n> MODEL_DIR=/path/to/Qwen3-Embedding-0.6B bash \
#     dev/issues/vllm-plugin-fl-pooling-nondeterminism/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/kunlun-infer-vllm/v1/env/v1"
CONTAINER=flagrt-kunlun-infer-v1-env-v1-p800a
AVOID_CARDS="${AVOID_CARDS:-1 2}"
IDLE_MIB_BAND=64
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${POOLING_COS_LOG_DIR:-/tmp/vllm-plugin-fl-pooling-nondeterminism-$TS}"
CASE_EVIDENCE="$HERE/assets/evidence"
RC_ENV=0; RC_RUN=0

die_usage() {
  echo "usage: XPU_CARD=<n> MODEL_DIR=/path/to/Qwen3-Embedding-0.6B bash $HERE/run.sh" >&2
  exit 2
}
[ -n "${XPU_CARD:-}" ] || die_usage
[ -n "${MODEL_DIR:-}" ] && [ -d "$MODEL_DIR" ] || die_usage
cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== vllm-plugin-fl-pooling-nondeterminism $(date -Is) card=${XPU_CARD} log=$LOG_DIR（收数，非判定）"

card_line_after() { grep -A1 -E "^\|[[:space:]]+$1[[:space:]]+P800" "$2" | sed -n 2p; }
card_used_mib() { card_line_after "$1" "$2" | grep -oE "[0-9]+MiB */ *[0-9]+MiB" | head -1 | cut -d/ -f1 | tr -d ' MiB'; }
card_util() { card_line_after "$1" "$2" | grep -oE "[0-9]+%" | head -1 | tr -d ' %'; }

xpu-smi > "$LOG_DIR/xpu-smi-before.txt" 2>&1 || { echo "[FAIL] xpu-smi 不可用"; exit 2; }
for a in $AVOID_CARDS; do
  [ "$a" = "$XPU_CARD" ] && { echo "[FAIL] 卡 $XPU_CARD 在 AVOID_CARDS（$AVOID_CARDS）"; exit 2; }
done
used="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
util="$(card_util "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
[ -n "$used" ] && [ -n "$util" ] || { echo "[FAIL] 卡 $XPU_CARD 未出现在 xpu-smi 输出"; exit 2; }
[ "$used" -gt "$IDLE_MIB_BAND" ] && { echo "[FAIL] 卡 $XPU_CARD 显存 ${used}MiB 高于空载带"; exit 2; }
[ "$util" -ne 0 ] && { echo "[FAIL] 卡 $XPU_CARD 利用率 ${util}% 非零"; exit 2; }
echo "[INFO] 卡 $XPU_CARD 空闲（显存 ${used}MiB，利用率 0%）"

read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
[ -n "$GOT_ID" ] && [ "$GOT_ID" = "$WANT_ID" ] || { echo "[FAIL] 镜像 id 与 env/v1 基线不一致"; exit 2; }
echo "[PASS] 镜像 id 与 env/v1 基线一致"

if ! XPU_CARD="$XPU_CARD" bash dev/lib/up.sh "$ENV_DIR" --strict; then
  echo "[FAIL] up.sh --strict 未通过"
  RC_ENV=2
fi

COS_DEFAULT=""; COS_NOCACHE=""
if [ "$RC_ENV" = "0" ]; then
  docker exec "$CONTAINER" mkdir -p /home/models/Qwen3-Embedding-0.6B
  docker cp "$MODEL_DIR/." "$CONTAINER:/home/models/Qwen3-Embedding-0.6B/" || RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # 【必要前提，不是本 case 的判定对象】容器现行镜像还没打包 AsyncModelRunnerOutput 修复
  # （见 ../vllm-plugin-fl-pooling-dispatch/），embed() 直接调用会先崩在那个断言上，走不到
  # 这里——本 case 关心的是"崩溃修掉之后，数值是否确定"，所以这里临时应用同一个已知的
  # 一行判断补丁（容器 down 后不持久，每次都要重打，这是正常现象）。这不代表 dispatch
  # 那个 case 也算验证过了——两个 case 分开记分。
  if ! docker exec "$CONTAINER" sed -i \
      's/if isinstance(output, (ModelRunnerOutput, NoneType)):/if isinstance(output, (ModelRunnerOutput, AsyncModelRunnerOutput, NoneType)):/' \
      /workspace/vllm-plugin-FL/vllm_fl/worker/worker.py; then
    echo "[FAIL] 应用 dispatch 前提补丁失败（容器内文件路径/行文本可能已变，检查镜像是否已升级）"
    RC_ENV=2
  fi
fi

if [ "$RC_ENV" = "0" ]; then
  PY_ENV=/root/miniconda/envs/python310_torch29_cuda/bin/python
  docker exec "$CONTAINER" mkdir -p /tmp/case-scripts
  docker cp "$HERE/assets/record_pooling_cos.py" "$CONTAINER:/tmp/case-scripts/record_pooling_cos.py"

  run_one() {  # $1 = 额外参数（如 --disable-prefix-caching），$2 = 日志文件名
    docker exec "$CONTAINER" bash -c "
        export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
        $PY_ENV /tmp/case-scripts/record_pooling_cos.py --model /home/models/Qwen3-Embedding-0.6B $1
      " > "$LOG_DIR/$2" 2>&1
  }
  run_one "" "cos-default.log"; RC1=$?
  COS_DEFAULT="$(grep -E '^COS = ' "$LOG_DIR/cos-default.log" | tail -1 | sed 's/COS = //')"
  run_one "--disable-prefix-caching" "cos-nocache.log"; RC2=$?
  COS_NOCACHE="$(grep -E '^COS = ' "$LOG_DIR/cos-nocache.log" | tail -1 | sed 's/COS = //')"
  cat "$LOG_DIR/cos-default.log" "$LOG_DIR/cos-nocache.log"
  [ "$RC1" = 0 ] && [ "$RC2" = 0 ] && RC_RUN=0 || RC_RUN=2
else
  RC_RUN=2
fi

python3 - "$CASE_EVIDENCE/summary.json" "${COS_DEFAULT:-null}" "${COS_NOCACHE:-null}" "$REF" "$GOT_ID" "$RC_RUN" <<'PYEOF'
import json, sys, datetime
out, cos_default, cos_nocache, ref, image_id, rc = sys.argv[1:7]
def f(v):
    try: return float(v)
    except ValueError: return None
json.dump({
    "schema": "flagrt.case-summary/v1",
    "case": "vllm-plugin-fl-pooling-nondeterminism",
    "executed_at": datetime.datetime.now().astimezone().isoformat(),
    "result": "NOT_APPLICABLE",
    "note": "收数脚本，根因未知，无验收判据；本字段不代表PASS/FAIL",
    "measurements": {
        "cos_prefix_caching_default_true": f(cos_default),
        "cos_prefix_caching_disabled": f(cos_nocache),
        "expected_if_deterministic": 1.0,
    },
    "exit_code": int(rc),
    "image": {"ref": ref, "id": image_id},
    "env": {"dir": "dev/images/kunlun-infer-vllm/v1/env/v1"},
}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PYEOF

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  XPU_CARD="$XPU_CARD" WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
sleep 8
xpu-smi > "$LOG_DIR/xpu-smi-after.txt" 2>&1 || true
used_after="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-after.txt")"
echo "[INFO] 收尾后卡 $XPU_CARD: 显存 ${used_after:-?}MiB"
[ -n "$used_after" ] && [ "$used_after" -gt "$IDLE_MIB_BAND" ] && echo "[FAIL] 卡 $XPU_CARD 显存未回落（${used_after}MiB）"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[FAIL] 容器 $CONTAINER 仍存在（残留）"
else
  echo "[PASS] 容器已删净（零残留）"
fi
echo "COS_DEFAULT=${COS_DEFAULT:-?} COS_NOCACHE=${COS_NOCACHE:-?}  完整日志: $LOG_DIR"
echo "本脚本不输出 overall PASS/FAIL（收数脚本，无验收判据）"
[ "$RC_ENV" = 0 ] && [ "$RC_RUN" = 0 ] && exit 0
exit 2
