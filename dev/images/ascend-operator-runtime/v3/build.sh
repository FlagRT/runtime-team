#!/usr/bin/env bash
# 重建 ascend-operator-runtime v3 —— Route A 默认血统 + 训练/推理统一基座。
#
# 【阶段声明】本脚本本身未在 phase 1 执行过——phase 1 明确禁止 docker build /
# docker run。写出本脚本是"起草"的一部分（与 v2 的 build.sh 保持同构，供 phase 2
# 直接拿去跑），不代表已验证可用。执行前请先看 REBUILD.md「开放问题」一节，
# 特别是 VLLM_ASCEND_REF 的精确 pin 是否已经核实过。
#
# 前置（准备构建上下文）：
#   src/FlagGems   git archive 自本机 checkout 的 pin commit（见 lock.yaml，与 v2 相同 commit）
#   src/Torch-FL   git archive 自本机 checkout 的 pin commit（见 lock.yaml，与 v2 相同 commit）
# llvm 工具链 / triton 编译依赖 / FlagTree 源码 / vllm-ascend 源码均由
# Dockerfile.repro 在构建期从公开资源现场拉取，不需要预置进构建上下文。
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
  -t flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 .

echo "校验:pip freeze 快照 -> assets/provenance/pipfreeze-operator-runtime-v3.txt（phase 2 补齐,当前目录只有 PENDING.md）"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 -m pip freeze | sort > /tmp/reproV3-pipfreeze.txt
echo "见 /tmp/reproV3-pipfreeze.txt"

echo "静态自检:"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 \
  /opt/flagrt/verify_runtime.py --static

echo "真机动态自检(需带 --device 容器,另见 REBUILD.md):"
echo "  python3 /opt/flagrt/verify_runtime.py            # 默认路径,预期 route_a_default_check: passed"
echo "  python3 /opt/flagrt/verify_runtime.py --check-torch-fl-guard"
