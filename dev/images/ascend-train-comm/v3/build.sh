#!/usr/bin/env bash
# 重建 ascend-train-comm v3 —— Route A / HCCL-adaptor 训练腿通信层。
# 父层 flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-... 需先构建
# (../ascend-operator-runtime/v3/build.sh)。
#
# 【阶段声明】本脚本本身未在 phase 1 执行过，见 REBUILD.md 顶部声明与
# 「开放问题」一节，执行前请先过一遍。
#
# 前置（准备构建上下文）：
#   src/FlagCX/   git archive 自本机 checkout 的 pin commit（见 lock.yaml，与 v2 相同 commit）
# third-party/json、third-party/googletest 由 Dockerfile.repro 在构建期从公开
# 上游 git clone，不需要预置进构建上下文。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${1:?用法: build.sh <构建上下文目录>（含 Dockerfile.repro + src/FlagCX）}"

cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
mkdir -p "$CTX/assets"
cp "$HERE/assets/verify_flagcx_runtime.py" "$HERE/assets/verify_flagcx_p2p.py" "$CTX/assets/"

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg PARENT_IMAGE=flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 \
  -t flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 .

echo "校验:pip freeze 快照 -> assets/provenance/pipfreeze-comm-v3.txt（phase 2 补齐,当前目录只有 PENDING.md）"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 \
  -m pip freeze | sort > /tmp/reproV3-comm-pipfreeze.txt
echo "见 /tmp/reproV3-comm-pipfreeze.txt"
echo "真机 2 卡验证(需带 --device 容器)另见 REBUILD.md，本脚本只做无卡构建。"
echo "  torchrun --nproc_per_node=2 /opt/flagrt/verify_flagcx_runtime.py --device"
echo "  torchrun --nproc_per_node=2 /opt/flagrt/verify_flagcx_p2p.py"
