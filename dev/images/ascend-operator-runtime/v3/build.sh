#!/usr/bin/env bash
# 重建 ascend-operator-runtime v3 —— Route A 默认血统 + 训练/推理统一基座。
# 已在 round 2–4 真机构建并验收(见 REBUILD.md / ARCHIVE.md)。
#
# 用法: build.sh [--prepare-only] <构建上下文目录>
#   构建上下文由本脚本自行准备:
#     src/FlagGems   <- 仓库根 FlagGems/          @ ARG FLAG_GEMS_COMMIT
#     src/Torch-FL   <- 仓库根 PyTorch-Plugin-FL/ @ ARG TORCH_FL_COMMIT
#   --prepare-only 只准备上下文,不 docker build。
# llvm 工具链 / triton 编译依赖 / FlagTree / vllm-plugin-FL 源码由
# Dockerfile.repro 在构建期从公开资源拉取,不需要预置进构建上下文。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
PREPARE_ONLY=0
if [[ "${1:-}" == "--prepare-only" ]]; then PREPARE_ONLY=1; shift; fi
CTX="${1:?用法: build.sh [--prepare-only] <构建上下文目录>}"
mkdir -p "$CTX"
CTX="$(cd "$CTX" && pwd)"

# 从同目录 Dockerfile.repro 的 ARG 默认值读 commit(唯一来源),git archive 到 $CTX/src/<name>。
prep_src() {  # prep_src <ARG 名> <仓库根下的 checkout 目录> <src 下的目标名>
  local arg="$1" repo="$REPO_ROOT/$2" dst="$CTX/src/$3" commit
  commit="$(sed -n "s/^ARG $arg=\([0-9a-f]\{40\}\)\$/\1/p" "$HERE/Dockerfile.repro")"
  [[ -n "$commit" ]] || { echo "错误: Dockerfile.repro 中未找到 ARG $arg=<40 位 commit>" >&2; exit 1; }
  if [[ -d "$dst" && -n "$(ls -A "$dst")" ]]; then
    echo "错误: $dst 已存在且非空,请先删除后重跑(避免混入旧上下文)" >&2; exit 1
  fi
  if ! git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$repo" fetch origin "$commit" || true
    git -C "$repo" cat-file -e "$commit^{commit}" 2>/dev/null \
      || { echo "错误: $repo 中不存在 commit $commit($arg),fetch 后仍缺失" >&2; exit 1; }
  fi
  mkdir -p "$dst"
  git -C "$repo" archive "$commit" | tar -x -C "$dst"
  echo "src/$3 <- $2@$commit"
}

prep_src FLAG_GEMS_COMMIT FlagGems FlagGems
prep_src TORCH_FL_COMMIT PyTorch-Plugin-FL Torch-FL
cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
mkdir -p "$CTX/assets"
cp "$HERE/assets/verify_runtime.py" \
   "$HERE/assets/patch_triton_ascend_flagtree.py" \
   "$HERE/assets/FlagGems-DSA-__init__.py" "$CTX/assets/"
# round 4（2026-10-08）：声明式覆盖层补丁——vllm-plugin-FL 上游 pin 之上应用
# assets/patches/vllm-plugin-FL/（决策①，详见该目录 README.md）。
mkdir -p "$CTX/assets/patches/vllm-plugin-FL"
cp "$HERE"/assets/patches/vllm-plugin-FL/*.patch "$CTX/assets/patches/vllm-plugin-FL/"
if (( PREPARE_ONLY )); then echo "构建上下文已就绪: $CTX"; exit 0; fi

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg BASE_IMAGE=harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-910c-py311-cann9.0.0-ubuntu22.04-aarch64:202608-torch2.10.0-vllm0.20.2 \
  -t flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 .

echo "校验:pip freeze 快照写到 /tmp,可与 assets/provenance/pipfreeze-operator-runtime-v3-round4*.txt 对比"
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
