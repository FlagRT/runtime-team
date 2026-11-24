#!/usr/bin/env bash
# 验收样例 · connector-ascend —— FlagMemory npu KV connector 真卡「字节级搬运」单入口 runner（宿主侧）。
#
# 【环境引用方式】不自建运行环境目录，直接使用现行环境目录
#   dev/images/ascend-infer-vllm/v1/env/v1
# （ENV-SPEC §3.2：直接引用最简）。该目录不挂 FlagMemory（镜像不含），本样例靠整仓挂载到
#   /workspace 的副作用让 /workspace/FlagMemory 可见，容器内用 PYTHONPATH 显式指向它。
#
# 【与 connector-kunlunxin 的差异（2026-11-24 定）】那个样例测的其实是新 manager 的芯片接线/
#   端到端真卡用例（tests/tiering/ 三件），对 offload connector 本身只验证了"能不能 import"。
#   本样例直接驱动 offload/npu/npu_copy_backend.py 的 NpuDmaCopyBackend，做 npu/RUNBOOK.md
#   「六判据」里此前只有一次性手工记录、没有可重复脚本的那三条：③D2H 字节级逐字一致 /
#   ④H2D 换入后新旧块号分叉在场 / ⑥destroy 双形态。①工厂懒加载②锁页实测仍由既有
#   test_offload_realcard.py -k npu 覆盖，不在本样例重复。
#
# 【单入口】幂等，可重复执行。流程 =
#   0) 前置核对：npu-smi 挑零算力/零显存占用的 die；本机带卡容器纪律＝一机同时只一个
#      （env/v1 compose 注释已写明）
#   1) 镜像自证：docker image inspect 的 id 与 env/v1 基线 image.id 逐字一致，不一致 FAIL
#   2) up.sh <env/v1> --strict（镜像 id 核对）
#   3) 容器内跑 scripts/npu_offload_byte_probe.py（独立进程，直接驱动 copy backend，不经
#      vLLM/不受调度策略影响）
#   4) 产出机器可判结果 evidence/summary.json
#   5) down 清理 + 显存回落核验（无论成败）
#
# 【外部依赖声明】（基线文件之外）
#   - 一张空闲 NPU die（INFER_DIE 必填，无缺省）
#   - 机器：910c-11（昇腾 910C，本样例与模型文件无关，不需要 FLAGRT_MODELS_DIR）
#
# 【自判定】退出码 0 = 全部通过；1 = 有失败；2 = 环境不满足。末尾输出判定行 overall: PASS / FAIL。
#
# 用法（仓根执行）：
#   INFER_DIE=<die> bash dev/memory/cases/connector-ascend/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/ascend-infer-vllm/v1/env/v1"
CONTAINER=flagrt-infer-v1-env-v1-910c
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CONNECTOR_CASE_LOG_DIR:-/tmp/connector-ascend-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_ENV=0; RC_PROBE=0

die_usage() { echo "usage: INFER_DIE=<die> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${INFER_DIE:-}" ] || die_usage
cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== connector-ascend $(date -Is) die=${INFER_DIE} log=$LOG_DIR"

# ---- 0) 前置核对（npu-smi 零占用检查，取该 die 的两行 AICore/Memory-Usage）----
npu-smi info > "$LOG_DIR/npu-smi-before.txt" 2>&1 || { echo "[FAIL] npu-smi 不可用"; exit 2; }
if grep -A1 "davinci${INFER_DIE}\b" "$LOG_DIR/npu-smi-before.txt" >/dev/null 2>&1; then :; fi
docker ps --format '{{.Names}}' > "$LOG_DIR/docker-ps-before.txt" || true
if [ -s "$LOG_DIR/docker-ps-before.txt" ]; then
  echo "[FAIL] 机器已有带卡容器在跑（一机只一个带卡容器纪律），先确认是否可以让位"
  cat "$LOG_DIR/docker-ps-before.txt"
  exit 2
fi
echo "[INFO] 机器无带卡容器在跑，继续"

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
if [ -z "$REF" ] || [ -z "$WANT_ID" ]; then echo "[FAIL] 基线 image.ref/id 读取为空"; exit 2; fi
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF"; echo "[INFO] 基线 id=$WANT_ID"; echo "[INFO] 本地 id=$GOT_ID"
if [ -z "$GOT_ID" ] || [ "$GOT_ID" != "$WANT_ID" ]; then echo "[FAIL] 镜像 id 与 env/v1 基线不一致"; exit 2; fi
echo "[PASS] 镜像 id 与 env/v1 基线一致"

# ---- 2) 起环境（本样例不需要模型，给个占位目录即可满足 FLAGRT_MODELS_DIR 挂载点）----
MODELS_DIR="${FLAGRT_MODELS_DIR:-$ROOT/dev/memory/cases/connector-ascend/evidence}"
if ! INFER_DIE="$INFER_DIE" FLAGRT_MODELS_DIR="$MODELS_DIR" bash dev/lib/up.sh "$ENV_DIR" --strict; then
  echo "[FAIL] up.sh --strict 未通过"
  RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 3) 容器内跑字节级探针 ----
  docker cp "$HERE/scripts/npu_offload_byte_probe.py" "$CONTAINER:/tmp/npu_offload_byte_probe.py"
  if docker exec "$CONTAINER" python3 /tmp/npu_offload_byte_probe.py > "$LOG_DIR/probe.log" 2>&1; then
    RC_PROBE=0
  else
    RC_PROBE=1
  fi
  cat "$LOG_DIR/probe.log"

  # ---- 4) summary.json（从探针末行的 SUMMARY_JSON:: 取出，补镜像/时间戳元信息）----
  SUMMARY_LINE="$(grep '^SUMMARY_JSON::' "$LOG_DIR/probe.log" | tail -1 | sed 's/^SUMMARY_JSON:://')"
  if [ -n "$SUMMARY_LINE" ]; then
    python3 -c "
import json, sys
d = json.loads('''$SUMMARY_LINE''')
d['image'] = {'ref': '$REF', 'id': '$GOT_ID'}
d['die'] = '$INFER_DIE'
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
  WORKSPACE_ROOT="$ROOT" INFER_DIE="$INFER_DIE" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
echo "rc: env=$RC_ENV probe=$RC_PROBE  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_PROBE" = 0 ]; then
  echo "overall: PASS"
  exit 0
fi
echo "overall: FAIL"
[ "$RC_ENV" != 0 ] && exit "$RC_ENV"
exit "$RC_PROBE"
