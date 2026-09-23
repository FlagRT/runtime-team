#!/usr/bin/env bash
# 校验 <docker-compose 文件...> 对应的 pins.<tag>.yaml 声明版本与宿主实际状态是否一致。
# 用法：verify_env.sh <docker-compose-file...>（与将要传给 docker compose -f 的文件列表一致）
# 不启动容器；只读校验 + 写 dev/lib/.env_provenance.json。
# 退出码：0=通过（无 strict/image 失败），1=有 strict/image 失败，2=用法或文件格式错误。
# 规范见 dev/ENV-SPEC.md。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$HERE/verify_env.py" "$@"
