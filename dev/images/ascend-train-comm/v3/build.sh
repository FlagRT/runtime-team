#!/usr/bin/env bash
# 重建 ascend-train-comm v3 —— Route A / HCCL-adaptor 训练腿通信层。
# 父层 flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-... 需先构建
# (../ascend-operator-runtime/v3/build.sh)。已在 round 2–4 真机构建并验收。
#
# 用法: build.sh [--prepare-only] <构建上下文目录>
#   构建上下文由本脚本自行准备:
#     src/FlagCX   <- 仓库根 FlagCX/ @ ARG FLAGCX_COMMIT
#   --prepare-only 只准备上下文,不 docker build。
# FlagCX 的 submodule(third-party/json、googletest)由 Dockerfile.repro 在构建期
# 从公开上游 git clone,git archive 不含 submodule 不影响构建。
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

prep_src FLAGCX_COMMIT FlagCX FlagCX
cp "$HERE/Dockerfile.repro" "$CTX/Dockerfile"
mkdir -p "$CTX/assets"
cp "$HERE/assets/verify_flagcx_runtime.py" "$HERE/assets/verify_flagcx_p2p.py" "$CTX/assets/"
if (( PREPARE_ONLY )); then echo "构建上下文已就绪: $CTX"; exit 0; fi

cd "$CTX"
DOCKER_BUILDKIT=0 docker build --network=host \
  --build-arg PARENT_IMAGE=flagrt/ascend-operator-runtime:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-arm64 \
  -t flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 .

echo "校验:pip freeze 快照写到 /tmp,可与 assets/provenance/pipfreeze-comm-v3-round4*.txt 对比"
docker run --rm --entrypoint /usr/local/python3.11.15/bin/python3 \
  flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64 \
  -m pip freeze | sort > /tmp/reproV3-comm-pipfreeze.txt
echo "见 /tmp/reproV3-comm-pipfreeze.txt"
echo "真机 2 卡验证(需带 --device 容器)另见 REBUILD.md，本脚本只做无卡构建。"
echo "  torchrun --nproc_per_node=2 /opt/flagrt/verify_flagcx_runtime.py --device"
echo "  torchrun --nproc_per_node=2 /opt/flagrt/verify_flagcx_p2p.py"
