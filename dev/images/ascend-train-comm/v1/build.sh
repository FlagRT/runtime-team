#!/usr/bin/env bash
# 重建 ascend-train-comm —— 功能等价重建（见 REBUILD.md）。
# 父层 flagrt/ascend-operator-runtime:0.2.0-reproB 需先构建（../ascend-operator-runtime/v1/build.sh）。
#
# 前置（准备构建上下文）：
#   flagcx-vendor/flagcx/               从原镜像 site-packages 回收的已装包
#   flagcx-vendor/flagcx-0.13.0.dist-info/  同上（metadata 供 importlib 识别）
#   （sha256 见 assets/provenance/flagcx-vendor.sha256；.so 与原镜像逐字节一致）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${1:?用法: build.sh <构建上下文目录>（含 Dockerfile.repro + flagcx-vendor/）}"

cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
cp "$HERE/assets/verify_flagcx_runtime.py" "$HERE/assets/verify_flagcx_p2p.py" "$CTX/"

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg PARENT_IMAGE=flagrt/ascend-operator-runtime:0.2.0-reproB \
  -t flagrt/ascend-operator-runtime-comm:0.1.3-reproB .

echo "校验：pip freeze 逐行一致 + flagcx .so sha256 一致"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime-comm:0.1.3-reproB -m pip freeze | sort | \
  diff - "$HERE/assets/provenance/GOLDEN-pipfreeze-comm.txt" && echo "PIP FREEZE OK"
