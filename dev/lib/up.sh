#!/usr/bin/env bash
# 启动运行环境目录：up.sh <env-dir> [--strict]
#
# 先 baseline.py check <env-dir>（结果写 <env-dir>/.last-check.json）：默认只报告
# 不一致、不阻止启动；加 --strict 时不一致即拒绝启动（用于验收与出结论的运行）。
# 然后 docker compose -f <env-dir>/docker-compose.yml up -d。
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

exec docker compose -f "$ENV_DIR/docker-compose.yml" up -d
