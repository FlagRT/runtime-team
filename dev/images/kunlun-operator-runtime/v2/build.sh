#!/usr/bin/env bash
# 重建 kunlun-operator-runtime v2。
#
# 用法: build.sh <构建上下文目录>
#   本层相对基座只新增 flagtree 一个包（FlagGems 直接用基座自带的那份，既不 COPY 也不重装），
#   因此构建上下文只有一个文件——从本目录拷进去的 Dockerfile。**重建不依赖宿主的任何
#   checkout**：不需要仓根 FlagGems/，也不需要任何 commit 已推送。
#
# 前置：
#   - 可拉取 harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base
#     （公开 harbor；本机已存在，image id ea6d797a7d44）
#   - 可访问 resource.flagos.net（flagtree wheel ~3.3GB）。**务必不经代理直连**：经
#     HTTP_PROXY/ALL_PROXY 转发实测反复 IncompleteRead 中断（见 REBUILD.md 踩坑 1）。
#     本脚本用 DOCKER_BUILDKIT=0 + --network=host 且不透传任何 *_proxy build-arg，
#     构建期容器内无代理环境变量，即为直连。
#   - /var/lib/docker 余量：**基座已在本机时留 ~12GB**。本层只给镜像加两个 layer，实测
#     （2026-10-11，docker history）flagtree 层 7.38GB（容器内 site-packages/triton 6.9GB）、
#     `rm -rf /root/.cache/pip` 层 0B，其余 ARG/LABEL/SHELL/WORKDIR 全为 0B；构建期另有
#     ~3.3GB 的 wheel 下载为瞬时占用。基座不在本机时另需为基座本体留出空间
#     （`docker images` 显示 105GB）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/root/miniconda/envs/python310_torch29_cuda/bin/python3

CTX="${1:?用法: build.sh <构建上下文目录>}"
mkdir -p "$CTX"
CTX="$(cd "$CTX" && pwd)"
cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  -t flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64 .

echo "校验：pip freeze 快照 -> assets/provenance/repro-pipfreeze-v2.txt"
docker run --rm --entrypoint "$PYTHON_BIN" \
  flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64 \
  -m pip freeze | sort \
  > "$HERE/assets/provenance/repro-pipfreeze-v2.txt"

echo "构建完成: flagrt/kunlun-operator-runtime:2.0.0-xpu3.6-py310-torch2.9-flagtree0.6.1-flaggems73c5aff1-x86_64"
echo "下一步（见 REBUILD.md / README.md）："
echo "  1) 生成并核对镜像基线："
echo "     python3 dev/lib/baseline.py export --image dev/images/kunlun-operator-runtime/v2 \\"
echo "       --demo dev/images/kunlun-operator-runtime/v2/demo/"
echo "     python3 dev/lib/baseline.py check  --image dev/images/kunlun-operator-runtime/v2"
echo "  2) 带卡自判定验证（先 docker ps + xpu-smi 看空闲卡，见 README「验证」）："
echo "     dev/lib/up.sh dev/images/kunlun-operator-runtime/v2/demo --strict"
