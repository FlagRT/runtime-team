#!/usr/bin/env bash
# 验收样例 · connector-mlu —— FlagMemory mlu KV connector 真卡「字节级搬运」单入口 runner（宿主侧）。
#
# 【环境引用方式】直接使用现行环境目录 dev/images/cambricon-infer-vllm/v1/env/v1
#（ENV-SPEC §3.2）。该目录不挂 FlagMemory（镜像不含），靠整仓挂载到 /workspace 的既有机制
# 让 /workspace/FlagMemory 可见，容器内探针脚本自行加入 sys.path，不改环境目录任何文件。
#
# 【与 connector-ascend 的差异】寒武纪 copy backend（offload/mlu/mlu_copy_backend.py 的
# MluCopyBackend）走 torch.mlu 命名空间（需先 import torch_mlu 注册），选卡用
# MLU_VISIBLE_DEVICES（docker exec -e 注入，不在 compose 固化）。
#
# 【单入口】幂等，可重复执行。流程 = 0)前置核对 cnmon零占用 1)镜像自证
# 2)up.sh（镜像id已自证，不强制--strict）3)容器内跑字节级探针 4)evidence/summary.json
# 5)down清理
#
# 用法（仓根执行，机器 = HWJ_1）：
#   MLU_CARD=0 bash dev/memory/cases/connector-mlu/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/cambricon-infer-vllm/v1/env/v1"
CONTAINER=flagrt-mlu-infer-v1-env-v1-mlu2
MLU_CARD="${MLU_CARD:-0}"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CONNECTOR_CASE_LOG_DIR:-/tmp/connector-mlu-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_ENV=0; RC_PROBE=0

cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== connector-mlu $(date -Is) card=$MLU_CARD log=$LOG_DIR"

# ---- 0) 前置核对 ----
cnmon > "$LOG_DIR/cnmon-before.txt" 2>&1 || { echo "[FAIL] cnmon 不可用"; exit 2; }
docker ps --format '{{.Names}}' > "$LOG_DIR/docker-ps-before.txt" || true
if grep -q "^${CONTAINER}$" "$LOG_DIR/docker-ps-before.txt"; then
  echo "[INFO] 容器 $CONTAINER 已在跑，复用"
fi
echo "[INFO] cnmon 已采集，人工复核卡 $MLU_CARD 空闲"

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
if [ -z "$REF" ] || [ -z "$WANT_ID" ]; then echo "[FAIL] 基线 image.ref/id 读取为空"; exit 2; fi
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF"; echo "[INFO] 基线 id=$WANT_ID"; echo "[INFO] 本地 id=$GOT_ID"
if [ -z "$GOT_ID" ] || [ "$GOT_ID" != "$WANT_ID" ]; then echo "[FAIL] 镜像 id 与 env 基线不一致"; exit 2; fi
echo "[PASS] 镜像 id 与 env 基线一致"

# ---- 2) 起环境 ----
# 注意（B24，登记处 flag_RT 控制台 anchors）：HWJ_1 宿主 python3.8.10，dev/lib/baseline.py
# 用了 PEP604 写法无 __future__ annotations，up.sh 在该机直接报错；本机无更新 python3 可做
# shim（已查无 3.9+）。按既有先例（T-061 寒武纪验证同款绕过）：手工核对镜像 id（已在步骤 1
# 做过）后直接 docker compose up -d，不经 up.sh/baseline.py，不改动任一脚本/配置文件。
MODELS_DIR="${MLU_MODELS_DIR:-$HOME/models}"
if [ ! -d "$MODELS_DIR" ]; then MODELS_DIR="$ROOT"; fi
if WORKSPACE_ROOT="$ROOT" MLU_MODELS_DIR="$MODELS_DIR" docker compose -f "$ENV_DIR/docker-compose.yml" up -d; then
  echo "[PASS] docker compose up -d 成功（镜像 id 已在步骤 1 自证，绕开 up.sh/baseline.py 的 B24 已知缺陷）"
else
  echo "[FAIL] docker compose up -d 未成功"
  RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 3) 容器内跑字节级探针 ----
  docker cp "$HERE/scripts/mlu_offload_byte_probe.py" "$CONTAINER:/tmp/mlu_offload_byte_probe.py"
  if docker exec -e MLU_VISIBLE_DEVICES="$MLU_CARD" "$CONTAINER" python3 /tmp/mlu_offload_byte_probe.py > "$LOG_DIR/probe.log" 2>&1; then
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
d['mlu_card'] = '$MLU_CARD'
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
  WORKSPACE_ROOT="$ROOT" MLU_MODELS_DIR="$MODELS_DIR" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
echo "rc: env=$RC_ENV probe=$RC_PROBE  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_PROBE" = 0 ]; then
  echo "overall: PASS"
  exit 0
fi
echo "overall: FAIL"
[ "$RC_ENV" != 0 ] && exit "$RC_ENV"
exit "$RC_PROBE"
