#!/usr/bin/env bash
# 重建 ascend-train-comm v1。需先备齐 lock.yaml:gaps 列出的资产：
#   assets/wheels/flagcx-0.13.0-cp311-cp311-linux_aarch64.whl
#   父镜像 flagrt/ascend-operator-runtime:0.2.0-...（或 dev/images/ascend-operator-runtime/v1/build.sh 先建）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="flagrt/ascend-operator-runtime-comm:0.1.3-cann9.0-py311-torch2.10-flagcx0.13.0g55eb2ffp2-arm64"
PARENT="${PARENT_IMAGE:-flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64}"

test -f assets/wheels/flagcx-0.13.0-cp311-cp311-linux_aarch64.whl || { echo "缺 flagcx wheel，见 lock.yaml:gaps"; exit 1; }
sha256sum -c assets/flagcx-wheel.sha256

DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  --build-arg PARENT_IMAGE="$PARENT" \
  -t "$TAG" .

echo "built $TAG"
echo "自检：docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 $TAG /opt/flagrt/verify_flagcx_runtime.py --static"
