#!/usr/bin/env bash
# 重建 ascend-operator-runtime v1。需先备齐 lock.yaml:gaps 列出的资产：
#   BASE_IMAGE（CANN 9.0.0 基座 sha256:a36a3022…）
#   assets/wheelhouse/  assets/mpich-4.1.3.tar.gz  assets/FlagGems-DSA-__init__.py
#   src/Torch-FL/（commit af50297…）  src/FlagGems/（commit f7ae8e6b…）
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TAG="flagrt/ascend-operator-runtime:0.2.0-cann9.0-py311-torch2.10-arm64"
BASE="${BASE_IMAGE:-flagrt/ascend-cann-base@sha256:a36a302204e282411dd46bd3f1edd64f7fce7190020bb4511e8b409ac8d4ba29}"

DOCKER_BUILDKIT=1 docker build \
  --platform linux/arm64 \
  --build-arg BASE_IMAGE="$BASE" \
  -t "$TAG" .

echo "built $TAG"
echo "自检：docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 $TAG /opt/flagrt/verify_runtime.py --static"
