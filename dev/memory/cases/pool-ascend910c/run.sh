#!/usr/bin/env bash
# 验收样例 · pool-ascend910c —— FlagMemory 池真卡用例（昇腾 910C）单入口 runner（宿主侧）。
#
# 【环境引用方式】本样例不自建运行环境目录，直接使用现行环境目录
#   dev/images/ascend-train-comm/v3/env/v2
# （ENV-SPEC §3.2：使用方可自建或直接引用；直接引用最简，且不产生第二份会漂移的装配——
#   版本事实唯一来源 = 该目录 env.baseline.yaml，由 up.sh --strict 机器核对）。
#
# 【单入口】幂等，可重复执行。流程 =
#   0) 前置核对：两张 die 显存≈驱动基线带（<4096MiB）且其 NPU 零计算进程；
#      本机带卡容器清点（空壳可共存：控制台 B3 判读＋2026-11-14 共存实证，如实记日志）
#   1) 镜像自证：docker image inspect 的 id 与 env/v2 基线 image.id 逐字一致，不一致 FAIL
#   2) up.sh <env/v2> --strict（镜像 id＋三挂载 commit＋cann-shmem 装后核对）
#   3) 容器内跑 FlagMemory tests/pool/test_pool_realcard.py 全量 11 用例
#      （口径按 FM 仓 tests/README：显式指到本文件＋FLAGMEMORY_ENABLE_LEGACY_CUDA=
#      I_ACCEPT_LEGACY_RISKS＋`-o addopts= -m realcard`；设备=容器内 0 号=TRAIN_DIE_A）
#   4) 产出机器可判结果 evidence/summary.json（用例名→pass/fail、计数、镜像 id、commit 锚、
#      cann-shmem 实装版本）
#   5) down 清理＋显存回落核验＋零残留（无论成败）
#
# 【外部依赖声明】（基线文件之外）
#   - pytest：容器内按需 pip 安装（python3 -m pip install pytest，默认 PyPI），与 FM 仓
#     dev extra 同源；本样例不固化其版本，只作测试运行器
#   - 两张空闲 die（TRAIN_DIE_A / TRAIN_DIE_B 必填，无缺省——每次执行前重新核对；
#     选卡纪律：npu-smi 挑显存为驱动基线且零进程者，优先未被并存容器映射的 die）
#   - 机器：910c-11（昇腾 910C，一机带卡容器纪律见现行锁定声明）
#
# 【自判定】退出码 0 = 全部通过（11/11）；1 = 有失败；2 = 环境不满足。末尾输出判定行
#   overall: PASS / FAIL。
#
# 【日志】完整日志落 /tmp/pool-ascend910c-<时间戳>/（容器侧经 /workspace/evidence/
#   pool-ascend910c/ 中转，该目录不入库）；仓内只留 evidence/summary.json 与 REPORT 截选。
#
# 用法（仓根执行）：
#   TRAIN_DIE_A=0 TRAIN_DIE_B=1 bash dev/memory/cases/pool-ascend910c/run.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
ENV_DIR="$ROOT/dev/images/ascend-train-comm/v3/env/v2"
CONTAINER=flagrt-train-v3-env-v2-910c
TS="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="${POOL_CASE_LOG_DIR:-/tmp/pool-ascend910c-$TS}"
CASE_EVIDENCE="$HERE/evidence"
RC_TESTS=0; RC_SUMMARY=0; RC_ENV=0

die_usage() { echo "usage: TRAIN_DIE_A=<die> TRAIN_DIE_B=<die> bash $HERE/run.sh" >&2; exit 2; }
[ -n "${TRAIN_DIE_A:-}" ] && [ -n "${TRAIN_DIE_B:-}" ] || die_usage
cd "$ROOT"
mkdir -p "$LOG_DIR" "$CASE_EVIDENCE"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1
echo "== pool-ascend910c $(date -Is) die=${TRAIN_DIE_A}/${TRAIN_DIE_B} log=$LOG_DIR"

# chip 行取法：`| <npu> <chip> | <busid> | <aicore> <mem>/<total> <hbm>/<total> |`
hbm_used_mib() {  # $1=die $2=file
  grep -E "^\|[[:space:]]*[0-9]+[[:space:]]+$1[[:space:]]+\|" "$2" | head -1 \
    | grep -oE '[0-9]+[[:space:]]*/[[:space:]]*[0-9]+' | tail -1 | cut -d/ -f1 | tr -d ' '
}
npu_of_die() { echo $(( $1 / 2 )); }   # 910C：一 NPU 两 die（0/1→NPU0，10/11→NPU5）
npu_has_process() {   # $1=npu $2=file ：进程表该 NPU 段内有无 pid 行
  sed -n "/^| NPU *Chip/,/^$/p" "$2" | grep -E "^\|[[:space:]]*$1[[:space:]]+[0-9]+[[:space:]]*\|[[:space:]]+[0-9]+" >/dev/null
}

# ---- 0) 前置核对 ----
npu-smi info > "$LOG_DIR/npu-smi-before.txt" 2>&1 || { echo "[FAIL] npu-smi 不可用"; exit 2; }
for d in "$TRAIN_DIE_A" "$TRAIN_DIE_B"; do
  used="$(hbm_used_mib "$d" "$LOG_DIR/npu-smi-before.txt")"
  if [ -z "$used" ]; then echo "[FAIL] die $d 未出现在 npu-smi 输出"; exit 2; fi
  if [ "$used" -gt 4096 ]; then
    echo "[FAIL] die $d HBM ${used}MiB 高于空载基线带（>4096），疑似被占用"; exit 2
  fi
  if npu_has_process "$(npu_of_die "$d")" "$LOG_DIR/npu-smi-before.txt"; then
    echo "[FAIL] die $d 所在 NPU 有计算进程"; exit 2
  fi
  echo "[INFO] die $d 空闲（HBM ${used}MiB，NPU 零进程）"
done
docker ps --format '{{.Names}}\t{{.Status}}' > "$LOG_DIR/docker-ps-before.txt" || true
for c in $(docker ps --format '{{.Names}}'); do
  if docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "$c" 2>/dev/null \
     | grep -q davinci; then
    echo "[INFO] 并存带卡容器: $c（空壳可共存，见 B3 判读；若其内有活跃 NPU 进程应停本样例）"
  fi
done

# ---- 1) 镜像自证 ----
read_ref() { python3 -c "import yaml;print(yaml.safe_load(open('$ENV_DIR/env.baseline.yaml'))['image']['$1'])"; }
REF="$(read_ref ref)"; WANT_ID="$(read_ref id)"
GOT_ID="$(docker image inspect "$REF" --format '{{.Id}}' 2>/dev/null || true)"
echo "[INFO] 镜像自证: ref=$REF"; echo "[INFO] 基线 id=$WANT_ID"; echo "[INFO] 本地 id=$GOT_ID"
if [ "$GOT_ID" != "$WANT_ID" ]; then echo "[FAIL] 镜像 id 与 env/v2 基线不一致"; exit 2; fi
echo "[PASS] 镜像 id 与 env/v2 基线一致"

# ---- 2) 起环境 ----
if ! TRAIN_DIE_A="$TRAIN_DIE_A" TRAIN_DIE_B="$TRAIN_DIE_B" \
     FLAGRT_MODELS_DIR="${FLAGRT_MODELS_DIR:-$ROOT/models}" \
     bash dev/lib/up.sh "$ENV_DIR" --strict; then
  echo "[FAIL] up.sh --strict 未通过（设备初始化失败时换空闲 die 重试一次；仍失败即停）"
  RC_ENV=2
fi

if [ "$RC_ENV" = "0" ]; then
  # ---- 3) 容器内跑 11 用例 ----
  docker exec "$CONTAINER" mkdir -p /workspace/evidence/pool-ascend910c
  if docker exec -e FLAGMEMORY_ENABLE_LEGACY_CUDA=I_ACCEPT_LEGACY_RISKS \
       -e FLAGMEMORY_REALCARD_DEVICE=0 \
       "$CONTAINER" bash /workspace/dev/memory/cases/pool-ascend910c/scripts/run_pool_realcard.sh; then
    RC_TESTS=0
  else
    RC_TESTS=1
  fi
  docker cp "$CONTAINER:/workspace/evidence/pool-ascend910c/junit.xml" "$LOG_DIR/junit.xml" >/dev/null 2>&1 || true
  docker cp "$CONTAINER:/workspace/evidence/pool-ascend910c/pytest.log" "$LOG_DIR/pytest.log" >/dev/null 2>&1 || true
  docker exec "$CONTAINER" python3 -m pip show cann-shmem > "$LOG_DIR/cann-shmem-version.txt" 2>&1 || true

  # ---- 4) summary.json ----
  if [ -f "$LOG_DIR/junit.xml" ]; then
    if python3 "$HERE/scripts/junit_to_summary.py" "$LOG_DIR/junit.xml" \
         "$ENV_DIR/env.baseline.yaml" "$CASE_EVIDENCE/summary.json" \
         "$LOG_DIR/cann-shmem-version.txt"; then
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
  TRAIN_DIE_A="$TRAIN_DIE_A" TRAIN_DIE_B="$TRAIN_DIE_B" \
    docker compose -f "$ENV_DIR/docker-compose.yml" down || true
fi
sleep 5
npu-smi info > "$LOG_DIR/npu-smi-after.txt" 2>&1 || true
for d in "$TRAIN_DIE_A" "$TRAIN_DIE_B"; do
  echo "[INFO] 收尾后 die $d: HBM $(hbm_used_mib "$d" "$LOG_DIR/npu-smi-after.txt")MiB"
done
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  echo "[FAIL] 容器 $CONTAINER 仍存在（残留）"
else
  echo "[PASS] 容器已删净（零残留）"
fi
echo "rc: env=$RC_ENV tests=$RC_TESTS summary=$RC_SUMMARY  完整日志: $LOG_DIR"
if [ "$RC_ENV" = 0 ] && [ "$RC_TESTS" = 0 ] && [ "$RC_SUMMARY" = 0 ]; then
  echo "overall: PASS"
  exit 0
fi
echo "overall: FAIL"
if [ "$RC_ENV" != 0 ]; then exit "$RC_ENV"; fi
[ "$RC_TESTS" != 0 ] && exit "$RC_TESTS"
exit "$RC_SUMMARY"
