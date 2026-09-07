#!/usr/bin/env bash
# ============================================================
# clone_all.sh —— 对齐 FlagRT 组织仓 6 个子库（新成员一键初始化 / 日常同步）
# 用法：./clone_all.sh [目标目录]；缺省目标 = 公共仓根（嵌套布局，子库收拢其内）
# 前置：SSH key 已上传且已是 FlagRT 组织成员（脚本自检，不过即退出）
#
# 分支纪律：本地只保留协作线，不保留主分支（main/master）——主分支仅以 origin/<名>
#   跟踪引用存在：Sync fork / merge origin/main 追上游照常，本地没有可误提交的主分支。
# 远端边界：对远端只做只读操作（clone/fetch），绝不 push 或改写远端任何状态；
#   全部写动作都在本地（切分支、快进合并、删除本地主分支）。
# ============================================================
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TARGET=${1:-$(dirname "$SCRIPT_DIR")}
mkdir -p "$TARGET"

# ---------- 认证预检（push 需要 SSH key，失败即退出） ----------
echo "[auth] 检查 GitHub SSH 认证 ..."
AUTH_OUT=$(ssh -T git@github.com 2>&1 || true)
if ! echo "$AUTH_OUT" | grep -q "successfully authenticated"; then
  echo "❌ SSH 认证失败。请先："
  echo "   1. 生成并上传 key：ssh-keygen -t ed25519 -C \"你的邮箱\" → GitHub Settings → SSH and GPG keys"
  echo "   2. 确认已是 FlagRT 组织成员（write 权限）"
  exit 1
fi
USER_NAME=$(echo "$AUTH_OUT" | sed -n 's/^Hi \([^!]*\)!.*/\1/p')
echo "[auth] 已认证账号: ${USER_NAME:-未知}"

# ---------- 仓库配置（唯一按仓维护处；新增/改名/换分支只改这里） ----------
# 每行一仓，| 分隔：URL | 本地目录名 | 协作线（检出/日常 pull） | 本地剔除的主分支（可空）
#   - 4 个开发仓：协作线 dev-1.0，同步分支 main
#   - FlagGems：同步分支是 master（fork 默认分支非 main），故剔除 master
#   - FlagTree：按上游 triton 版本线维护，协作线 triton_v3.2.x
#   - FlagPerf：无 dev-1.0，main 即唯一工作线 → 剔除列留空（例外保留本地 main）
# 注：PyTorch-Plugin-FL 目录名沿用上游改名（Torch-FL）前的容器挂载约定
#     （/workspace/PyTorch-Plugin-FL 不变），故 URL 与目录名分离、各自独立。
# 字段用 | 分隔是为兼容 macOS 自带 Bash 3.2（无关联数组）。
REPO_CONF=(
  "git@github.com:FlagRT/Torch-FL.git|PyTorch-Plugin-FL|dev-1.0|main"
  "git@github.com:FlagRT/FlagCX.git|FlagCX|dev-1.0|main"
  "git@github.com:FlagRT/FlagGems.git|FlagGems|dev-1.0|master"
  "git@github.com:FlagRT/vllm-plugin-FL.git|vllm-plugin-FL|dev-1.0|main"
  "git@github.com:FlagRT/FlagTree.git|FlagTree|triton_v3.2.x|main"
  "git@github.com:FlagRT/FlagPerf.git|FlagPerf|main|"
)

# ---------- 通用工具 ----------
is_primary() { # $1=主分支名单（空格分隔） $2=分支名
  local p
  for p in $1; do [ "$p" = "$2" ] && return 0; done
  return 1
}

# 剔除本地主分支：仅无独有提交（可安全 -d）时删，有独有提交则保留并警告，绝不 -D 强删
purge_primaries() { # $1=仓目录 $2=当前分支 $3=主分支名单
  local dir=$1 cur=$2 p
  for p in $3; do
    [ "$p" = "$cur" ] && continue
    git -C "$dir" show-ref --verify --quiet "refs/heads/$p" || continue
    if git -C "$dir" branch -d "$p" >/dev/null 2>&1; then
      echo "    - 已删除本地 $p（保留 origin/$p 跟踪引用，追上游不受影响）"
    else
      echo "    - ⚠️  本地 $p 未删除：含未合入 origin/$p 的独有提交等，请人工处理（勿 -D 强删）"
    fi
  done
}

# 对齐单个仓库（6 仓共用；差异全部收敛在 REPO_CONF）
align_repo() { # $1=URL $2=目录名 $3=协作线 $4=主分支名单
  local url=$1 dir="$TARGET/$2" work=$3 primaries=$4
  echo "== $2（origin=$url，协作线 $work）=="
  if [ -d "$dir/.git" ]; then
    echo "[sync] fetch + 对齐 ..."
    # fetch --prune 先行：切协作线（DWIM）、-d 安全性判定都依赖最新 origin/*；仅读远端
    if ! git -C "$dir" fetch origin --prune; then
      echo "    ⚠️  fetch 失败，跳过 $2"
      return
    fi
    cur=$(git -C "$dir" symbolic-ref --short -q HEAD || true)
    if [ "$cur" = "$work" ]; then
      git -C "$dir" merge --ff-only "origin/$work" \
        || echo "    ⚠️  $work 快进失败（本地有未推送独有提交？）"
    elif [ -n "$cur" ] && is_primary "$primaries" "$cur"; then
      # 当前站在主分支上 → 切到协作线再快进
      if git -C "$dir" checkout "$work" 2>/dev/null; then
        echo "    - 已从本地 $cur 切到协作线 $work"
        git -C "$dir" merge --ff-only "origin/$work" || echo "    ⚠️  $work 快进失败"
      else
        echo "    ⚠️  切到 $work 失败（有未提交改动？），仍停留 $cur；本地主分支未删"
      fi
    else
      # 个人开发分支：原地 pull（无 upstream/本地领先会失败，警告即可，不擅动用户分支）
      git -C "$dir" pull --ff-only 2>/dev/null \
        || echo "    ⚠️  pull 失败（个人分支无 upstream 或本地有未推送改动？），跳过 $2 对齐"
    fi
    cur=$(git -C "$dir" symbolic-ref --short -q HEAD || true)
    purge_primaries "$dir" "$cur" "$primaries"
  else
    echo "[clone] 协作线 $work ..."
    if ! git clone -b "$work" "$url" "$dir"; then
      echo "❌ 克隆 $2 失败：确认已是 FlagRT 组织成员（private 仓需成员权限）"
      exit 1
    fi
    if [ -n "$primaries" ]; then
      echo "    - 本地分支仅 $work（未创建本地 $primaries）"
    else
      echo "    - 本地分支仅 $work（该仓协作线即主分支，例外保留）"
    fi
    echo "[ok] $2 origin=FlagRT"
  fi
}

# ---------- 主流程 ----------
for conf in "${REPO_CONF[@]}"; do
  IFS='|' read -r url dir work primaries <<< "$conf"
  align_repo "$url" "$dir" "$work" "$primaries"
done

echo
echo "完成。当前各仓本地分支："
for conf in "${REPO_CONF[@]}"; do
  IFS='|' read -r url dir work primaries <<< "$conf"
  repo="$TARGET/$dir"
  [ -d "$repo/.git" ] && echo "  $dir: $(git -C "$repo" branch --format='%(refname:short)' | tr '\n' ' ')"
done
echo
echo "下一步："
echo "  1. 建个人开发分支再动手：cd <子库> && git checkout -b <名字>/<功能>（禁止裸 checkout main/master）"
echo "  2. 起容器：cd dev/<子方向> && docker compose -f ../compose.base.yml -f docker-compose.yml up -d"
echo "  3. 容器内验证：ls /workspace 应看到 6 个 repo"
