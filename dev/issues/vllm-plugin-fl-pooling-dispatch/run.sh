#!/usr/bin/env bash
# vllm-plugin-fl-pooling-dispatch —— 单入口 runner（宿主侧）。
#
# 【环境引用方式】不自建运行环境目录，直接使用现行环境目录
#   dev/images/kunlun-infer-vllm/v1/env/v1
# （版本事实唯一来源 = 该目录 env.baseline.yaml，由 up.sh --strict 机器核对）。
#
# 【单入口】幂等，可重复执行，不修改容器内任何文件（不打补丁——补丁是验证用的临时
# 手段，不是环境的永久状态；本脚本只检测"当前这份环境的 embed() 还崩不崩"，用于
# README「版本集合」的分版表态）。流程 =
#   0) 前置核对：xpu-smi 挑零负载/零显存卡；避开 AVOID_CARDS
#   1) 镜像自证：docker image inspect 的 id 与 env/v1 基线 image.id 逐字一致，不一致 FAIL
#   2) up.sh <env/v1> --strict
#   3) docker cp 模型目录进容器（基线文件之外的依赖，见 README 前置条件表）
#   4) 容器内跑 assets/minirepro_pooling_dispatch.py（脚本自带三态判定）
#   5) 产出机器可判结果 evidence/summary.json
#   6) down 清理＋显存回落核验＋零残留（无论成败）
#
# 【外部依赖声明】（基线文件之外）
#   - 一张空闲 P800 卡（XPU_CARD 必填，无缺省）
#   - MODEL_DIR：宿主侧 Qwen3-Embedding-0.6B 本地路径（已剔除
#     config_sentence_transformers.json / modules.json），必填，无缺省
#
# 【自判定】退出码 0 = overall PASS（缺陷不存在/已修）；1 = FAIL（缺陷复现）；
#   2 = ERROR（环境不满足或撞到另一个问题）。
#
# 【日志】完整日志落 /tmp/vllm-plugin-fl-pooling-dispatch-<时间戳>/；仓内只留
#   evidence/summary.json 与 README 里的截选（assets/evidence/*.txt）。
#
# 用法（仓根执行）：
#   XPU_CARD=<n> MODEL_DIR=/path/to/Qwen3-Embedding-0.6B bash \
#     dev/issues/vllm-plugin-fl-pooling-dispatch/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/kunlun-infer-vllm/v1/env/v1"
CONTAINER=flagrt-kunlun-infer-v1-env-v1-p800a
AVOID_CARDS="${AVOID_CARDS:-1 2}"
IDLE_MIB_BAND=64
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${POOLING_DISPATCH_LOG_DIR:-/tmp/vllm-plugin-fl-pooling-dispatch-$TS}"
CASE_EVIDENCE="$HERE/assets/evidence"
RC_ENV=0; RC_RUN=0; RC_SUMMARY=0

die_usage() {
  echo "usage: XPU_CARD=<n> MODEL_DIR=/path/to/Qwen3-Embedding-0.6B bash $HERE/run.sh" >&2
  exit 2
}
[ -n "${XPU_CARD:-}" ] || die_usage
[ -n "${MODEL_DIR:-}" ] && [ -d "$MODEL_DIR" ] || die_usage
cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== vllm-plugin-fl-pooling-dispatch $(date -Is) card=${XPU_CARD} log=$LOG_DIR"

card_line_after() { grep -A1 -E "^\|[[:space:]]+$1[[:space:]]+P800" "$2" | sed -n 2p; }
card_used_mib() { card_line_after "$1" "$2" | grep -oE "[0-9]+MiB */ *[0-9]+MiB" | head -1 | cut -d/ -f1 | tr -d ' MiB'; }
card_util() { card_line_after "$1" "$2" | grep -oE "[0-9]+%" | head -1 | tr -d ' %'; }

# ---- 0) 前置核对 ----
xpu-smi > "$LOG_DIR/xpu-smi-before.txt" 2>&1 || { echo "[FAIL] xpu-smi 不可用"; exit 2; }
for a in $AVOID_CARDS; do
  [ "$a" = "$XPU_CARD" ] && { echo "[FAIL] 卡 $XPU_CARD 在 AVOID_CARDS（$AVOID_CARDS），换卡执行"; exit 2; }
done
used="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
util="$(card_util "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
[ -n "$used" ] && [ -n "$util" ] || { echo "[FAIL] 卡 $XPU_CARD 未出现在 xpu-smi 输出"; exit 2; }
[ "$used" -gt "$IDLE_MIB_BAND" ] && { echo "[FAIL] 卡 $XPU_CARD 显存 ${used}MiB 高于空载带，疑似被占用"; exit 2; }
[ "$util" -ne 0 ] && { echo "[FAIL] 卡 $XPU_CARD 利用率 ${util}% 非零"; exit 2; }
echo "[INFO] 卡 $XPU_CARD 空闲（显存 ${used}MiB，利用率 0%）"

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
[ -n "$REF" ] && [ -n "$WANT_ID" ] || { echo "[FAIL] 基线 image.ref/id 读取为空"; exit 2; }
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF 基线id=$WANT_ID 本地id=$GOT_ID"
[ -n "$GOT_ID" ] && [ "$GOT_ID" = "$WANT_ID" ] || { echo "[FAIL] 镜像 id 与 env/v1 基线不一致"; exit 2; }
echo "[PASS] 镜像 id 与 env/v1 基线一致"

# ---- 2) 起环境 ----
if ! XPU_CARD="$XPU_CARD" bash dev/lib/up.sh "$ENV_DIR" --strict; then
  echo "[FAIL] up.sh --strict 未通过"
  RC_ENV=2
fi

VERDICT_LINE=""
if [ "$RC_ENV" = "0" ]; then
  # ---- 3) 模型目录（基线外依赖，docker cp 进去——容器 down 后不持久，每次都要重 cp）----
  docker exec "$CONTAINER" mkdir -p /home/models/Qwen3-Embedding-0.6B
  if ! docker cp "$MODEL_DIR/." "$CONTAINER:/home/models/Qwen3-Embedding-0.6B/"; then
    echo "[FAIL] 模型 docker cp 失败"
    RC_ENV=2
  fi
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 4) 跑最小复现（脚本自带 export 三个网络变量的前置检查 + 自判定）----
  # 脚本内部以 Path(__file__) 重新 spawn 自己做子进程隔离，不能经 stdin 注入
  # （那样 __file__ 会落在 /dev/stdin，子进程重跑不到自己）——须先 docker cp 落盘。
  PY_ENV=/root/miniconda/envs/python310_torch29_cuda/bin/python
  docker exec "$CONTAINER" mkdir -p /tmp/case-scripts
  docker cp "$HERE/assets/minirepro_pooling_dispatch.py"     "$CONTAINER:/tmp/case-scripts/minirepro_pooling_dispatch.py"
  if docker exec "$CONTAINER" bash -c "
      export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo VLLM_HOST_IP=127.0.0.1
      $PY_ENV /tmp/case-scripts/minirepro_pooling_dispatch.py --model /home/models/Qwen3-Embedding-0.6B
    " > "$LOG_DIR/repro.log" 2>&1; then
    RC_RUN=0
  else
    RC_RUN=$?
  fi
  cat "$LOG_DIR/repro.log"
  VERDICT_LINE="$(grep -E '^overall: ' "$LOG_DIR/repro.log" | tail -1)"
else
  RC_RUN=2
fi

# ---- 5) summary.json ----
RESULT="ERROR"
case "$VERDICT_LINE" in
  "overall: PASS") RESULT="PASS" ;;
  "overall: FAIL") RESULT="FAIL" ;;
esac
python3 - "$CASE_EVIDENCE/summary.json" "$RESULT" "$REF" "$GOT_ID" "$RC_RUN" "$LOG_DIR" <<'PYEOF'
import json, sys, datetime
out, result, ref, image_id, rc, log_dir = sys.argv[1:7]
json.dump({
    "schema": "flagrt.case-summary/v1",
    "case": "vllm-plugin-fl-pooling-dispatch",
    "executed_at": datetime.datetime.now().astimezone().isoformat(),
    "result": result,
    "criteria": {
        "script": "assets/minirepro_pooling_dispatch.py",
        "expect_when_fixed": "overall: PASS, EMBED_OK dim=1024",
    },
    "exit_code": int(rc),
    "image": {"ref": ref, "id": image_id},
    "env": {"dir": "dev/images/kunlun-infer-vllm/v1/env/v1"},
    "full_log": f"{log_dir}/repro.log（不入库，机器名见执行记录）",
}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PYEOF
RC_SUMMARY=$?

# ---- 6) 收尾（无论成败）----
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
echo "rc: env=$RC_ENV run=$RC_RUN summary=$RC_SUMMARY  完整日志: $LOG_DIR"
echo "overall: $RESULT"
case "$RESULT" in
  PASS) exit 0 ;;
  FAIL) exit 1 ;;
  *) exit 2 ;;
esac
