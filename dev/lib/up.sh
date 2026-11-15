#!/usr/bin/env bash
# 启动运行环境目录：up.sh <env-dir> [--strict]
#
# 先 baseline.py check <env-dir>（结果写 <env-dir>/.last-check.json）：默认只报告
# 不一致、不阻止启动；加 --strict 时不一致即拒绝启动（用于验收与出结论的运行）。
# 然后 docker compose -f <env-dir>/docker-compose.yml up -d。
# 起容器后按基线 env_packages 登记幂等安装运行环境层包（ENV-SPEC §3.3；无登记时无操作，
# 旧环境行为不变）；--strict 装完再 check 一次，此时容器在跑，env_packages 节为实测核对。
# 规范见 dev/ENV-SPEC.md。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { echo "usage: up.sh <env-dir> [--strict]" >&2; exit 2; }

[ "$#" -ge 1 ] && [ -d "$1" ] || usage
ENV_DIR="$(cd "$1" && pwd)"
shift
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --strict) STRICT=1 ;;
    *) usage ;;
  esac
done

# 单一来源：compose 的 WORKSPACE_ROOT 与 baseline.py 的 FLAGRT_WORKSPACE_ROOT 取同一值，
# 保证 check 核对的就是容器实际挂载的 checkout。
WORKSPACE_ROOT="${FLAGRT_WORKSPACE_ROOT:-$(cd "$HERE/../.." && pwd)}"
export WORKSPACE_ROOT FLAGRT_WORKSPACE_ROOT="$WORKSPACE_ROOT"

set +e
python3 "$HERE/baseline.py" check "$ENV_DIR"
RC=$?
set -e
if [ "$RC" -ge 2 ]; then
  echo "[FAIL] 基线核对无法执行（用法或文件格式错误），拒绝启动" >&2
  exit "$RC"
fi
if [ "$RC" -ne 0 ]; then
  if [ "$STRICT" -eq 1 ]; then
    echo "[FAIL] --strict：基线核对不一致，拒绝启动" >&2
    exit "$RC"
  fi
  echo "[WARN] 基线核对不一致：默认只报告、不阻止启动（验收/出结论的运行请加 --strict）" >&2
fi

docker compose -f "$ENV_DIR/docker-compose.yml" up -d

# 环境层包（ENV-SPEC §3.3）：起容器后按基线登记幂等安装；已装且版本一致则跳过。
# 安装失败（含容器在等待期退出/超时）即失败退出，不留半装配环境。
set +e
python3 "$HERE/baseline.py" install "$ENV_DIR"
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
  echo "[FAIL] env_packages 安装失败（详见上方 baseline.py install 输出）" >&2
  exit "$RC"
fi

# --strict 装完复核：此时容器已运行，check 的 env_packages 节为容器内实测（§3.3 规则 3），
# 其余节（image id / 挂载 repo）与启动前同一口径，幂等再核一遍。
if [ "$STRICT" -eq 1 ]; then
  set +e
  python3 "$HERE/baseline.py" check "$ENV_DIR"
  RC=$?
  set -e
  if [ "$RC" -ne 0 ]; then
    echo "[FAIL] --strict：装后基线核对不一致" >&2
    exit "$RC"
  fi
fi
