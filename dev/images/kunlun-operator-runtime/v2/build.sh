#!/usr/bin/env bash
# 重建 kunlun-operator-runtime v2。
#
# 用法: build.sh [--prepare-only] <构建上下文目录>
#   构建上下文由本脚本自行准备（与 ascend-train-comm/v3/build.sh 同构）：
#     src/FlagGems  <- 仓库根 FlagGems/ @ ARG FLAG_GEMS_COMMIT（git archive，只含已提交内容）
#   --prepare-only 只准备上下文，不 docker build。
#
# 前置：
#   - 可拉取 harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base
#     （公开 harbor；本机已存在，image id ea6d797a7d44）
#   - 可访问 resource.flagos.net（flagtree wheel ~3.3GB）。**务必不经代理直连**：经
#     HTTP_PROXY/ALL_PROXY 转发实测反复 IncompleteRead 中断（见 REBUILD.md 踩坑 2）。
#     本脚本用 DOCKER_BUILDKIT=0 + --network=host 且不透传任何 *_proxy build-arg，
#     构建期容器内无代理环境变量，即为直连。
#   - 被 pin 的 FlagGems commit 须已推送到远端，否则他人无法从零重建（本脚本会检查）。
#   - /var/lib/docker 需 ~15GB 余量（本层增量约 11GB，多数是 flagtree wheel 解包后的体积）。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
PYTHON_BIN=/root/miniconda/envs/python310_torch29_cuda/bin/python3

PREPARE_ONLY=0
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=1; shift; fi
CTX="${1:?用法: build.sh [--prepare-only] <构建上下文目录>}"
mkdir -p "$CTX"
CTX="$(cd "$CTX" && pwd)"

# 从同目录 Dockerfile.repro 的 ARG 默认值读 commit（唯一来源），git archive 到 $CTX/src/<name>。
prep_src() {  # prep_src <ARG 名> <仓库根下的 checkout 目录> <src 下的目标名>
  local arg="$1" repo="$REPO_ROOT/$2" dst="$CTX/src/$3" commit
  commit="$(sed -n "s/^ARG $arg=\([0-9a-f]\{40\}\)\$/\1/p" "$HERE/Dockerfile.repro")"
  [[ -n "$commit" ]] || { echo "错误: Dockerfile.repro 中未找到 ARG $arg=<40 位 commit>" >&2; exit 1; }
  if [[ -d "$dst" && -n "$(ls -A "$dst")" ]]; then
    echo "错误: $dst 已存在且非空，请先删除后重跑（避免混入旧上下文）" >&2; exit 1
  fi
  if ! git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$repo" fetch origin "$commit" || true
    git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null \
      || { echo "错误: $repo 中不存在 commit $commit（$arg），fetch 后仍缺失" >&2; exit 1; }
  fi
  if [[ -z "$(git -C "$repo" branch -r --contains "$commit" 2>/dev/null)" ]]; then
    echo "错误: $2@$commit 不在任何远端分支上，他人无法从零重建（先 push 该 commit）" >&2; exit 1
  fi
  mkdir -p "$dst"
  git -C "$repo" archive "$commit" | tar -x -C "$dst"
  echo "src/$3 <- $2@$commit"
}

prep_src FLAG_GEMS_COMMIT FlagGems FlagGems
cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
if (( PREPARE_ONLY )); then echo "构建上下文已就绪: $CTX"; exit 0; fi

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
