#!/usr/bin/env bash
# 校验通过后再启动：<docker-compose 文件...> 与传给 docker compose -f 的文件列表一致。
# 用法：up.sh <docker-compose-file...>
# 总组发版与子方向预研用同一个入口，区别只在传入几层文件、每层写了什么。
# 规范见 dev/ENV-SPEC.md。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
  echo "usage: up.sh <docker-compose-file...>" >&2
  exit 2
fi

"$HERE/verify_env.sh" "$@"

export WORKSPACE_ROOT="$(cd "$HERE/../.." && pwd)"

ARGS=()
for f in "$@"; do
  ARGS+=(-f "$f")
done
docker compose "${ARGS[@]}" up -d
