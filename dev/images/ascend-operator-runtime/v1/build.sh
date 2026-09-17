#!/usr/bin/env bash
# 重建 ascend-operator-runtime —— 功能等价重建（见 REBUILD.md）。
#
# 前置（准备构建上下文）：
#   assets/wheelhouse/          从原镜像 /opt/flagrt/wheelhouse 回收（或按 assets/wheelhouse.sha256 重建）
#   assets/mpich-4.1.3.tar.gz   MPICH 项目官方源码发行版，sha256 见 assets/mpich-4.1.3.tar.gz.sha256
#   src/FlagGems  src/Torch-FL  git archive 自本机 checkout 的 pin commit（见 lock.yaml）
#   CPython 3.11.15             Dockerfile.repro 构建期从 python.org 拉取（需 --network=host）
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${1:?用法: build.sh <构建上下文目录>（含 Dockerfile.repro + wheelhouse/ + mpich tar + src/）}"

cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
cp "$HERE/assets/requirements-runtime.txt" "$HERE/assets/verify_runtime.py" \
   "$HERE/assets/patch_triton_ascend_3_2_1.py" "$HERE/assets/FlagGems-DSA-__init__.py" "$CTX/"

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg BASE_IMAGE=harbor.baai.ac.cn/flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev \
  -t flagrt/ascend-operator-runtime:0.2.0-reproB .

echo "校验：pip freeze 应与 assets/provenance/GOLDEN-pipfreeze-operator-runtime.txt 逐行一致"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime:0.2.0-reproB -m pip freeze | sort | \
  diff - "$HERE/assets/provenance/GOLDEN-pipfreeze-operator-runtime.txt" && echo "PIP FREEZE OK"
