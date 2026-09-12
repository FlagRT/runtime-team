#!/usr/bin/env bash
# 重建 ascend-operator-runtime v2 —— 全新血统首次构建(见 REBUILD.md)。
#
# 前置(准备构建上下文):
#   src/FlagGems   git archive 自本机 checkout 的 pin commit(见 lock.yaml)
#   src/Torch-FL   git archive 自本机 checkout 的 pin commit(见 lock.yaml)
# llvm 工具链 / triton 编译依赖 / FlagTree 源码本身由 Dockerfile.repro 在构建期
# 从 BAAI·FlagTree 官方公开资源现场拉取,不需要预置进构建上下文。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${1:?用法: build.sh <构建上下文目录>（含 Dockerfile.repro + src/FlagGems + src/Torch-FL）}"

cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
mkdir -p "$CTX/assets"
cp "$HERE/assets/verify_runtime.py" \
   "$HERE/assets/patch_triton_ascend_flagtree.py" \
   "$HERE/assets/FlagGems-DSA-__init__.py" "$CTX/assets/"

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg BASE_IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2 \
  -t flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64 .

echo "校验:pip freeze 快照(参照 assets/provenance/pipfreeze-operator-runtime-v2.txt,血统不同不要求逐行一致)"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime:1.0.0-flagtree3.5-cann9.0-py311-torch2.10-arm64 -m pip freeze | sort > /tmp/reproV2-pipfreeze.txt
echo "见 /tmp/reproV2-pipfreeze.txt"
