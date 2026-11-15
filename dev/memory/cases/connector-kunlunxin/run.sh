#!/usr/bin/env bash
# 验收样例 · connector-kunlunxin —— FlagMemory xpu KV connector 真卡用例（昆仑芯 P800）单入口 runner（宿主侧）。
#
# 【环境引用方式】本样例不自建运行环境目录，直接使用现行环境目录
#   dev/images/kunlun-infer-vllm/v1/env/v1
# （ENV-SPEC §3.2：直接引用最简，不产生第二份会漂移的装配；版本事实唯一来源 = 该目录
#   env.baseline.yaml，由 up.sh --strict 机器核对）。
#
# 【单入口】幂等，可重复执行。流程 =
#   0) 前置核对：xpu-smi 挑零负载/零显存卡；避开 AVOID_CARDS（默认"1 2"，分配挂死前科）
#      与有残留占用的卡；本机带卡容器清点（映射全 8 卡的空壳可共存，以 xpu-smi 实测为准）
#   1) 镜像自证：docker image inspect 的 id 与 env/v1 基线 image.id 逐字一致，不一致 FAIL
#   2) up.sh <env/v1> --strict（镜像 id + FlagMemory 挂载 commit 核对）
#   3) 容器内核 FlagMemory 版本（import flagmemory == 0.3.0，PYTHONPATH 生效口径）
#   4) 容器内跑 connector 判据三件（RUNBOOK 口径，tests/tiering/ 三个 realcard 文件，共 5 用例）：
#      · test_offload_realcard.py -k xpu（栈可导入取到导出类）×1
#      · test_xpu_backend_realcard.py（工厂懒加载/锁页实测/D2H·H2D 逐字/换入分叉源行判据）×1
#      · test_xpu_tier_manager_realcard.py（T8 负例/异步轨全链=审计三等/destroy 双形态）×3
#   5) 产出机器可判结果 evidence/summary.json（用例名→pass/fail、计数、镜像 id、FM commit 锚）
#   6) down 清理＋显存回落核验＋零残留（无论成败）
#
# 【外部依赖声明】（基线文件之外）
#   - 一张空闲 P800 卡（XPU_CARD 必填，无缺省——每次执行前重新核对；镜像 conda env 自带
#     pytest，无需网络/pip；判据三件运行无需模型目录）
#   - 机器：P800-A（昆仑芯 P800 ×8，选卡纪律见 env/v1 compose 头注）
#
# 【自判定】退出码 0 = 全部通过（5/5）；1 = 有失败；2 = 环境不满足。末尾输出判定行
#   overall: PASS / FAIL。
#
# 【日志】完整日志落 /tmp/connector-kunlunxin-<时间戳>/（容器侧经 /tmp/case-evidence/ 中转，
#   不入库）；仓内只留 evidence/summary.json 与 REPORT 截选。
#
# 用法（仓根执行）：
#   XPU_CARD=<n> bash dev/memory/cases/connector-kunlunxin/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/kunlun-infer-vllm/v1/env/v1"
CONTAINER=flagrt-kunlun-infer-v1-env-v1-p800a
AVOID_CARDS="${AVOID_CARDS:-1 2}"
IDLE_MIB_BAND=64
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${CONNECTOR_CASE_LOG_DIR:-/tmp/connector-kunlunxin-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_ENV=0; RC_VER=0; RC_TESTS=0; RC_SUMMARY=0

die_usage() { echo "usage: XPU_CARD=<n> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${XPU_CARD:-}" ] || die_usage
cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== connector-kunlunxin $(date -Is) card=${XPU_CARD} log=$LOG_DIR"

# xpu-smi 卡行取法：卡行 `| <id>  P800 OAM ...`，紧随行含 `<used>MiB / <total>MiB`（首个
# MiB 对 = 设备显存）与 `<n>%`（利用率）。
card_line_after() {  # $1=card $2=file
  grep -A1 -E "^\|[[:space:]]+$1[[:space:]]+P800" "$2" | sed -n 2p
}
card_used_mib() {    # $1=card $2=file
  card_line_after "$1" "$2" | grep -oE "[0-9]+MiB */ *[0-9]+MiB" | head -1 | cut -d/ -f1 | tr -d ' MiB'
}
card_util() {        # $1=card $2=file
  card_line_after "$1" "$2" | grep -oE "[0-9]+%" | head -1 | tr -d ' %'
}

# ---- 0) 前置核对 ----
xpu-smi > "$LOG_DIR/xpu-smi-before.txt" 2>&1 || { echo "[FAIL] xpu-smi 不可用"; exit 2; }
for a in $AVOID_CARDS; do
  if [ "$a" = "$XPU_CARD" ]; then
    echo "[FAIL] 卡 $XPU_CARD 在 AVOID_CARDS（$AVOID_CARDS，分配挂死前科），换卡执行"; exit 2
  fi
done
used="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
util="$(card_util "$XPU_CARD" "$LOG_DIR/xpu-smi-before.txt")"
if [ -z "$used" ] || [ -z "$util" ]; then
  echo "[FAIL] 卡 $XPU_CARD 未出现在 xpu-smi 输出"; exit 2
fi
if [ "$used" -gt "$IDLE_MIB_BAND" ]; then
  echo "[FAIL] 卡 $XPU_CARD 显存 ${used}MiB 高于空载带（>${IDLE_MIB_BAND}MiB），疑似被占用"; exit 2
fi
if [ "$util" -ne 0 ]; then
  echo "[FAIL] 卡 $XPU_CARD 利用率 ${util}% 非零，有在跑负载"; exit 2
fi
echo "[INFO] 卡 $XPU_CARD 空闲（显存 ${used}MiB，利用率 0%）"
docker ps --format '{{.Names}}\t{{.Status}}' > "$LOG_DIR/docker-ps-before.txt" || true
for c in $(docker ps --format '{{.Names}}'); do
  if docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" 2>/dev/null \
     | grep -q "xpu${XPU_CARD}\b"; then
    echo "[INFO] 并存容器 $c 亦映射卡 $XPU_CARD（空壳可共存；若其内有活跃负载，上方占用核对会拦下）"
  fi
done

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
if [ -z "$REF" ] || [ -z "$WANT_ID" ]; then echo "[FAIL] 基线 image.ref/id 读取为空"; exit 2; fi
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF"; echo "[INFO] 基线 id=$WANT_ID"; echo "[INFO] 本地 id=$GOT_ID"
if [ -z "$GOT_ID" ] || [ "$GOT_ID" != "$WANT_ID" ]; then echo "[FAIL] 镜像 id 与 env/v1 基线不一致"; exit 2; fi
echo "[PASS] 镜像 id 与 env/v1 基线一致"

# ---- 2) 起环境 ----
if ! XPU_CARD="$XPU_CARD" bash dev/lib/up.sh "$ENV_DIR" --strict; then
  echo "[FAIL] up.sh --strict 未通过（设备初始化失败时换空闲卡重试一次；仍失败即停）"
  RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 3) FlagMemory 版本核对 ----
  PY_ENV=/root/miniconda/envs/python310_torch29_cuda/bin/python
  if docker exec "$CONTAINER" $PY_ENV -c "import flagmemory,sys; v=flagmemory.__version__; print('FM_VERSION', v); sys.exit(0 if v=='0.3.0' else 1)"; then
    echo "[PASS] 容器内 flagmemory 0.3.0（PYTHONPATH 生效）"
  else
    echo "[FAIL] 容器内 flagmemory 版本非 0.3.0（或导入失败）"
    RC_VER=1
  fi

  # ---- 4) 判据三件（脚本经 stdin 注入；容器不挂仓根，见 env/v1 compose 头注）----
  if docker exec -i "$CONTAINER" bash -s < "$HERE/scripts/run_connector_realcard.sh"; then
    RC_TESTS=0
  else
    RC_TESTS=1
  fi
  for f in fm-version.txt junit-offload.xml junit-backend.xml junit-tiermgr.xml \
           pytest-offload.log pytest-backend.log pytest-tiermgr.log; do
    docker cp "$CONTAINER:/tmp/case-evidence/$f" "$LOG_DIR/$f" >/dev/null 2>&1 || true
  done

  # ---- 5) summary.json ----
  junit_files=""
  for f in "$LOG_DIR"/junit-*.xml; do [ -f "$f" ] && junit_files="$junit_files $f"; done
  if [ -n "$junit_files" ]; then
    if python3 "$HERE/scripts/junit_to_summary.py" "$ENV_DIR/env.baseline.yaml" \
         "$CASE_EVIDENCE/summary.json" "$LOG_DIR/fm-version.txt" $junit_files; then
      RC_SUMMARY=0
    else
      RC_SUMMARY=1
    fi
  else
    echo "[FAIL] 未取得 junit xml（测试未产出结果）"
    RC_SUMMARY=1
  fi
fi

# ---- 6) 收尾（无论成败）----
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  XPU_CARD="$XPU_CARD" WORKSPACE_ROOT="$ROOT" docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
sleep 8
xpu-smi > "$LOG_DIR/xpu-smi-after.txt" 2>&1 || true
used_after="$(card_used_mib "$XPU_CARD" "$LOG_DIR/xpu-smi-after.txt")"
echo "[INFO] 收尾后卡 $XPU_CARD: 显存 ${used_after:-?}MiB"
if [ -n "$used_after" ] && [ "$used_after" -gt "$IDLE_MIB_BAND" ]; then
  echo "[FAIL] 卡 $XPU_CARD 显存未回落（${used_after}MiB > ${IDLE_MIB_BAND}MiB）"
fi
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[FAIL] 容器 $CONTAINER 仍存在（残留）"
else
  echo "[PASS] 容器已删净（零残留）"
fi
echo "rc: env=$RC_ENV ver=$RC_VER tests=$RC_TESTS summary=$RC_SUMMARY  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_VER" = 0 ] && [ "$RC_TESTS" = 0 ] && [ "$RC_SUMMARY" = 0 ]; then
  echo "overall: PASS"
  exit 0
fi
echo "overall: FAIL"
[ "$RC_ENV" != 0 ] && exit "$RC_ENV"
[ "$RC_VER" != 0 ] && exit "$RC_VER"
[ "$RC_TESTS" != 0 ] && exit "$RC_TESTS"
exit "$RC_SUMMARY"
