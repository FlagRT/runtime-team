#!/usr/bin/env bash
# ============================================================
# clone_all.sh —— 对齐 dev/repos.yaml 登记的全部仓（组织子库 + 方向交付仓；新成员一键初始化 / 日常同步）
# 用法：./clone_all.sh [目标目录]；缺省目标 = 公共仓根（嵌套布局，各仓收拢其内）
# 前置：SSH key 已上传且已是 FlagRT 组织成员（脚本自检，不过即退出）；python3 + yaml（解析登记文件）
#
# 登记处：dev/repos.yaml（SSOT，报备-收录制）；新增/改名/换协作线只改 yaml，不改本脚本。
#   无权限的 private 仓降级跳过并提示向总组申请权限，不使整体失败。
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

# ---------- 仓库配置（唯一登记处：dev/repos.yaml；新增/改名/换协作线只改 yaml） ----------
# 本脚本零硬编码，逐仓字段全部来自 repos.yaml；解析为 | 分隔行：
#   URL | 本地目录名 | 协作线（检出/日常 pull） | 本地剔除的主分支（空格分隔，可空） | 可见性
# 只对齐布局与协作线、不锁提交：版本事实由运行环境 env.baseline.yaml 的 mounted_repos 记录。
# 字段用 | 分隔、按行读入普通数组，是为兼容 macOS 自带 Bash 3.2（无关联数组）。
REPOS_YAML="$SCRIPT_DIR/repos.yaml"
# 解析器先读入变量再 python3 -c 执行：避开 Bash 3.2 对 $(...) 内嵌 heredoc 的解析缺陷
read -r -d '' PARSE_PY <<'PY' || true
import sys
try:
    import yaml
except ImportError:
    sys.exit("缺少 python3 yaml 模块（pip install pyyaml）")
path = sys.argv[1]
try:
    with open(path) as f:
        data = yaml.safe_load(f)
except FileNotFoundError:
    sys.exit("登记文件不存在：%s" % path)
except yaml.YAMLError as e:
    sys.exit("yaml 语法错误：%s\n%s" % (path, e))
if not isinstance(data, dict) or data.get("schema") != "flagrt.repos/v1":
    sys.exit("schema 不是 flagrt.repos/v1：%s" % path)
repos = data.get("repos")
if not isinstance(repos, list) or not repos:
    sys.exit("repos 列表缺失或为空：%s" % path)
seen = set()
for i, r in enumerate(repos):
    if not isinstance(r, dict):
        sys.exit("repos[%d] 不是映射" % i)
    for k in ("url", "dir", "branch", "kind", "owner", "visibility", "purge_primaries"):
        if k not in r or r[k] is None or (k != "purge_primaries" and str(r[k]).strip() == ""):
            sys.exit("repos[%d] 缺字段 %s" % (i, k))
    if r["kind"] not in ("sublib", "direction"):
        sys.exit("repos[%d] kind 非法：%s" % (i, r["kind"]))
    if r["visibility"] not in ("public", "private"):
        sys.exit("repos[%d] visibility 非法：%s" % (i, r["visibility"]))
    if not isinstance(r["purge_primaries"], list):
        sys.exit("repos[%d] purge_primaries 须为列表" % i)
    if r["dir"] in seen:
        sys.exit("dir 重复：%s" % r["dir"])
    seen.add(r["dir"])
    vals = [str(r["url"]), str(r["dir"]), str(r["branch"]),
            " ".join(str(p) for p in r["purge_primaries"]), r["visibility"]]
    if any("|" in v or "\n" in v for v in vals):
        sys.exit("repos[%d] 字段含 | 或换行" % i)
    print("|".join(vals))
PY
if ! CONF_OUT=$(python3 -c "$PARSE_PY" "$REPOS_YAML"); then
  echo "❌ 解析 dev/repos.yaml 失败（见上方报错），未对任何仓做改动"
  exit 1
fi
REPO_CONF=()
while IFS= read -r line; do
  [ -n "$line" ] && REPO_CONF+=("$line")
done <<< "$CONF_OUT"
echo "[conf] 已从 dev/repos.yaml 读入 ${#REPO_CONF[@]} 仓"

# ---------- 认证预检（push 需要 SSH key，失败即退出） ----------
echo "[auth] 检查 GitHub SSH 认证 ..."
# SSH 主机取自登记文件首仓 url 的 user@host 部分（脚本不硬编码地址）
AUTH_HOST=${REPO_CONF[0]%%:*}
AUTH_OUT=$(ssh -T "$AUTH_HOST" 2>&1 || true)
if ! echo "$AUTH_OUT" | grep -q "successfully authenticated"; then
  echo "❌ SSH 认证失败。请先："
  echo "   1. 生成并上传 key：ssh-keygen -t ed25519 -C \"你的邮箱\" → GitHub Settings → SSH and GPG keys"
  echo "   2. 确认已是 FlagRT 组织成员（write 权限）"
  exit 1
fi
USER_NAME=$(echo "$AUTH_OUT" | sed -n 's/^Hi \([^!]*\)!.*/\1/p')
echo "[auth] 已认证账号: ${USER_NAME:-未知}"

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

# private 仓访问失败的提示（Repository not found / permission denied 多为未开通权限）
hint_private() { # $1=可见性 $2=目录名
  [ "$1" = "private" ] || return 0
  echo "    ↳ $2 为 private 仓：若报 Repository not found / permission denied，"
  echo "      请向总组申请访问权限（报备时登记的成员名单）"
}

# 对齐单个仓库（各仓共用；差异全部收敛在 dev/repos.yaml）
align_repo() { # $1=URL $2=目录名 $3=协作线 $4=主分支名单 $5=可见性
  local url=$1 dir="$TARGET/$2" work=$3 primaries=$4 vis=$5
  echo "== $2（origin=$url，协作线 $work）=="
  if [ -d "$dir/.git" ]; then
    echo "[sync] fetch + 对齐 ..."
    # fetch --prune 先行：切协作线（DWIM）、-d 安全性判定都依赖最新 origin/*；仅读远端
    if ! git -C "$dir" fetch origin --prune; then
      echo "    ⚠️  fetch 失败，跳过 $2"
      hint_private "$vis" "$2"
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
      if [ "$vis" = "private" ]; then
        echo "    ⚠️  克隆 $2 失败，跳过"
        hint_private "$vis" "$2"
        return
      fi
      echo "❌ 克隆 $2 失败：确认已是 FlagRT 组织成员"
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
  IFS='|' read -r url dir work primaries vis <<< "$conf"
  align_repo "$url" "$dir" "$work" "$primaries" "$vis"
done

echo
echo "完成。当前各仓本地分支："
for conf in "${REPO_CONF[@]}"; do
  IFS='|' read -r url dir work primaries vis <<< "$conf"
  repo="$TARGET/$dir"
  if [ -d "$repo/.git" ]; then
    echo "  $dir: $(git -C "$repo" branch --format='%(refname:short)' | tr '\n' ' ')"
  else
    echo "  $dir: （未就位，见上方提示）"
  fi
done
echo
echo "下一步："
echo "  1. 建个人开发分支再动手：cd <子库> && git checkout -b <名字>/<功能>（禁止裸 checkout main/master）"
echo "  2. 起容器：cd dev/<子方向> && docker compose -f ../compose.base.yml -f docker-compose.yml up -d"
echo "  3. 容器内验证：ls /workspace 应看到 dev/repos.yaml 登记的各仓（无权限跳过的除外）"
