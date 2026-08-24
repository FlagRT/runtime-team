#!/usr/bin/env bash
# 对齐 FlagRT 组织仓 6 个仓库到最新状态（团队新成员/日常同步用）
# 说明：本脚本位于 runtime-team 公共仓 dev/ 内；新成员已 clone 本仓，
#       运行脚本把其余 6 个子库对齐：缺失 → clone，已存在 → 更新到最新。
# 前置（认证在脚本外完成）：
#   1) SSH key 已上传到 GitHub（Settings → SSH and GPG keys）
#   2) 已是 FlagRT 组织成员（write 权限，见 README 协作纪律）
# 用法：./clone_all.sh [目标目录，默认公共仓根目录（嵌套布局：子库收拢在公共仓内）]
# 职责：仅预检认证 + 对齐子库；不接触任何凭据
set -euo pipefail

# 默认目标 = 公共仓根目录 = 开发总目录（脚本位于公共仓的 dev/ 下，其上级即公共仓根；嵌套布局）
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=${1:-$(dirname "$SCRIPT_DIR")}
mkdir -p "$TARGET"

# ---------- 认证预检（push 需要 SSH key，失败即退出） ----------
echo "[auth] 检查 GitHub SSH 认证 ..."
AUTH_OUT=$(ssh -T git@github.com 2>&1 || true)
if ! echo "$AUTH_OUT" | grep -q "successfully authenticated"; then
  echo "❌ SSH 认证失败。请先："
  echo "   1. 生成并上传 key：ssh-keygen -t ed25519 -C \"你的邮箱\""
  echo "      → GitHub Settings → SSH and GPG keys → New SSH key → 粘贴 .pub 内容"
  echo "   2. 确认已是 FlagRT 组织成员"
  exit 1
fi
USER_NAME=$(echo "$AUTH_OUT" | sed -n 's/^Hi \([^!]*\)!.*/\1/p')
echo "[auth] 已认证账号: ${USER_NAME:-未知}"

# ---------- 对齐 6 仓（origin = FlagRT 组织仓） ----------
# 使用普通数组 + case，兼容 macOS 自带 Bash 3.2；新增仓库需同时在 branch_for 中声明基线分支。
REPOS=(PyTorch-Plugin-FL FlagCX FlagGems vllm-plugin-FL FlagTree FlagPerf)

branch_for() {
  case "$1" in
    PyTorch-Plugin-FL|FlagCX|vllm-plugin-FL|FlagPerf) echo main ;;
    FlagGems) echo master ;;
    FlagTree) echo triton_v3.2.x ;;
    *) echo "❌ 未声明基线分支: $1" >&2; return 1 ;;
  esac
}

# 上游/组织已改名 Torch-FL；本地目录沿用 PyTorch-Plugin-FL（容器路径约定 /workspace/PyTorch-Plugin-FL）。
remote_for() {
  case "$1" in
    PyTorch-Plugin-FL) echo Torch-FL ;;
    *) echo "$1" ;;
  esac
}

for name in "${REPOS[@]}"; do
  dir="$TARGET/$name"
  branch=$(branch_for "$name")
  remote_name=$(remote_for "$name")
  if [ -d "$dir/.git" ]; then
    echo "[sync] $name 更新到最新（ff-only）..."
    if ! git -C "$dir" pull --ff-only; then
      echo "⚠️   $name 更新失败：本地有未提交/未推送改动？请先 commit 或 stash 后重试"
    fi
    continue
  fi
  echo "[clone] $name (分支 $branch) ..."
  if ! git clone -b "$branch" "git@github.com:FlagRT/${remote_name}.git" "$dir"; then
    echo "❌ 克隆 $name 失败：确认已是 FlagRT 组织成员（private 仓需成员权限）"
    exit 1
  fi
  echo "[ok]   $name  origin=FlagRT"
done

echo
echo "完成。下一步："
echo "  1. 进入协作开发分支：各仓分支见 VERSIONS.md §1（FlagTree 用 triton_v3.2.x）"
echo "  2. 起容器：cd dev/<子方向> && docker compose -f ../compose.base.yml -f docker-compose.yml up -d"
echo "     （当前已有子方向：memory、communication；各子方向容器配置见 dev/ 下对应目录）"
echo "  3. 容器内验证：ls /workspace 应看到 6 个 repo"
echo "  4. 探针：见公共仓 dev/memory/README.md（probes/ 待入库）"
