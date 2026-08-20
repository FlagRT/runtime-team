#!/usr/bin/env bash
# dev-prep.sh — 开发前置一键准备(对齐 clone_all.sh 的 5 子库 + 公共仓 runtime-team 自身)
# 位置:公共仓 dev/memory/ 下(memory 子方向内部使用);默认目标 = 公共仓根(脚本目录上两级,嵌套布局)
#
# 作用:
#   1) 每个仓 git fetch origin --prune —— 本地 refs/remotes/origin/* 全部更新到远端最新
#      (dev-1.0 / main / master / triton_v3.2.x 等所有远端分支引用,prune 顺手清掉远端已删的)
#   2) 每个仓切到本地开发分支 <用户名>/dev(下称 DEV_BRANCH,默认取当前系统用户):
#        · 纯本地分支,从不绑定远端上游
#        · 不存在 → 从远端最新开发线创建(优先 origin/dev-1.0;FlagTree 无 dev-1.0,
#                    回退 origin/triton_v3.2.x)
#        · 已存在 → 切过去,ff-only 对齐基线;若本地有独有提交则保留并提示
#                   (开发分支不丢你已写的代码);旧版脚本误设的上游会被自动解绑
#        · --reset → 强制硬对齐基线(reset --hard,丢弃该分支独有提交,慎用)
#   3) 本地已有的 dev-1.0/main/master/triton_v3.2.x 真实分支默认同步到远端最新
#      (ff-only:无独有提交才推进,分叉分支保留并提示,不产生合并提交;可用 --no-align 关闭)
#
# 用法:
#   ./dev-prep.sh [目标目录] [--reset] [--no-align] [--auto-stash] [--dry-run]
#     目标目录: 默认公共仓根目录(嵌套布局,与 clone_all.sh 一致)
#     --reset:      DEV_BRANCH 强制硬对齐到最新基线(reset --hard,丢弃其独有提交,慎用)
#     --no-align:   不同步本地 dev-1.0/main/master/triton_v3.2.x 真实分支
#                   (默认会同步:ff-only 推进,分叉保留,不碰远端)
#     --auto-stash: 工作区有未提交改动时自动 stash 并事后 pop;默认遇到脏工作区跳过该仓
#     --dry-run:    只打印每个仓将要执行的动作,不执行任何写操作
#   分支名: 默认 <当前系统用户名>/dev(如 xliu969/dev);可用环境变量 DEV_BRANCH 覆盖
#
# 说明:本脚本只动本地分支与远端跟踪引用,不 push、不创建/合并 PR,不触碰远端分支。

# ---------- 解析参数 ----------
RESET=0; ALIGN=1; AUTO_STASH=0; DRY_RUN=0
POS=()
for a in "$@"; do
  case "$a" in
    --reset)      RESET=1 ;;
    --no-align)   ALIGN=0 ;;
    --auto-stash) AUTO_STASH=1 ;;
    --dry-run)    DRY_RUN=1 ;;
    *)            POS+=("$a") ;;
  esac
done
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# 脚本位于 dev/memory/ 下:上两级 = 公共仓根;若脚本移动位置需同步调整层级
TARGET=${POS[0]:-$(dirname "$(dirname "$SCRIPT_DIR")")}
[ -d "$TARGET" ] || { echo "❌ 目标目录不存在: $TARGET"; exit 1; }

# 本地开发分支名:<用户名>/dev(可 DEV_BRANCH 环境变量覆盖)
DEV_BRANCH="${DEV_BRANCH:-${USER:-$(whoami)}/dev}"

# ---------- 仓库清单 ----------
# 与 clone_all.sh 对齐:5 子库 + 公共仓自身(嵌套布局,子库收拢在公共仓根下)
REPOS=(PyTorch-Plugin-FL FlagCX FlagGems vllm-plugin-FL FlagTree runtime-team)
# 无 dev-1.0 的仓 → 回退到各自主线
declare -A FALLBACK_BASE=(
  [FlagTree]=origin/triton_v3.2.x
)

[ "$DRY_RUN" = 1 ] && echo "▶ DRY-RUN 模式:只打印计划,不执行"
echo "▶ 目标目录: $TARGET"
echo "▶ 开发分支: $DEV_BRANCH  (创建时基线: 优先 origin/dev-1.0,FlagTree 回退 triton_v3.2.x)"
echo

# ---------- 单个仓库处理 ----------
prep_repo() {
  local name="$1" dir="$2"
  echo "── $name ──"
  [ -d "$dir/.git" ] || { echo "  [skip] 未找到 $dir/.git,跳过"; return; }

  local base="origin/dev-1.0"
  [ -n "${FALLBACK_BASE[$name]:-}" ] && base="${FALLBACK_BASE[$name]}"
  # fetch 后校验基线引用存在(防止远端结构变化);不存在则退回 origin/HEAD
  if [ "$DRY_RUN" = 0 ] && ! git -C "$dir" rev-parse --verify --quiet "refs/remotes/$base" >/dev/null 2>&1; then
    echo "  [warn] 远端引用 $base 不存在,回退 origin/HEAD"
    base="origin/HEAD"
  fi

  # 1) fetch
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [dry] git fetch origin --prune"
  else
    echo "  [git] fetch origin --prune …"
    git -C "$dir" fetch origin --prune >/dev/null 2>&1 \
      || echo "  [warn] fetch 失败(网络/认证?),继续尝试分支操作"
  fi

  # 2) 工作区干净检查
  local STASHED=0
  if [ "$DRY_RUN" = 0 ] && ! git -C "$dir" diff --quiet; then
    if [ "$AUTO_STASH" = 1 ]; then
      echo "  [git] 工作区有改动,自动 stash …"
      git -C "$dir" stash push -m "dev-prep-$(date +%s)" >/dev/null
      STASHED=1
    else
      echo "  [skip] 工作区有未提交改动,跳过本仓(先 commit,或加 --auto-stash)"
      return
    fi
  fi

  # 3) DEV_BRANCH 分支(纯本地分支,从不绑定远端)
  if [ "$DRY_RUN" = 1 ]; then
    if git -C "$dir" show-ref --verify --quiet "refs/heads/$DEV_BRANCH"; then
      echo "  [dry] checkout $DEV_BRANCH(已存在)"
      [ "$RESET" = 1 ] && echo "  [dry] --reset: reset --hard $base"
    else
      echo "  [dry] git checkout -b $DEV_BRANCH $base  (纯本地,不设上游)"
    fi
  else
    if git -C "$dir" show-ref --verify --quiet "refs/heads/$DEV_BRANCH"; then
      echo "  [git] checkout $DEV_BRANCH …"
      git -C "$dir" checkout "$DEV_BRANCH" >/dev/null 2>&1 || { echo "  [warn] checkout 失败,跳过本仓"; return; }
      # 旧版脚本误设过上游:解绑,保持"纯本地分支"约定
      if git -C "$dir" rev-parse --verify --quiet "@{upstream}" >/dev/null 2>&1; then
        echo "  [git] 解绑上游(纯本地分支约定)…"
        git -C "$dir" branch --unset-upstream >/dev/null 2>&1 || true
      fi
      if [ "$RESET" = 1 ]; then
        echo "  [git] --reset: reset --hard $base"
        git -C "$dir" reset --hard "$base" >/dev/null 2>&1
      else
        echo "  [git] ff-only 对齐基线 $base …"
        git -C "$dir" merge --ff-only "$base" >/dev/null 2>&1 \
          || echo "  [warn] 本地有独有提交,已保留(可手动 git merge $base 或加 --reset)"
      fi
    else
      echo "  [git] 新建 $DEV_BRANCH ← $base(纯本地,不设上游)…"
      git -C "$dir" checkout -b "$DEV_BRANCH" "$base" >/dev/null 2>&1
      # 不设上游;若 git 因同名分支自动跟踪,立即解绑
      git -C "$dir" branch --unset-upstream >/dev/null 2>&1 || true
    fi
  fi

  # 4) 本地真实分支同步(默认开;ff-only:可快进才推进,分叉保留提示)
  if [ "$ALIGN" = 1 ] && [ "$DRY_RUN" = 1 ]; then
    echo "  [dry] 同步本地 dev-1.0/main/master/triton_v3.2.x(ff-only;--no-align 关闭)"
  elif [ "$ALIGN" = 1 ]; then
    echo "  [git] 同步本地 dev-1.0/main/master/triton_v3.2.x(ff-only)…"
    for b in dev-1.0 main master triton_v3.2.x; do
      git -C "$dir" show-ref --verify --quiet "refs/heads/$b" || continue
      git -C "$dir" rev-parse --verify --quiet "refs/remotes/origin/$b" >/dev/null 2>&1 || continue
      [ "$(git -C "$dir" rev-parse "$b")" = "$(git -C "$dir" rev-parse "origin/$b")" ] && continue
      if git -C "$dir" merge-base --is-ancestor "$b" "origin/$b" >/dev/null 2>&1; then
        git -C "$dir" branch -f "$b" "origin/$b"
        echo "  [git] 本地 $b → 对齐 origin/$b"
      else
        echo "  [warn] 本地 $b 有独有提交,未对齐(需手动处理)"
      fi
    done
  fi

  # 5) stash 还原
  if [ "$STASHED" = 1 ]; then
    echo "  [git] stash pop …"
    git -C "$dir" stash pop >/dev/null 2>&1 \
      || echo "  [warn] stash pop 冲突,请手动处理(git stash list)"
  fi

  # 结果
  if [ "$DRY_RUN" = 0 ]; then
    echo "  [ok]   当前分支: $(git -C "$dir" branch --show-current)"
    echo "  [info] 本地分支绑定状态(fetch 已更新 origin/* 引用):"
    git -C "$dir" branch -vv | grep -v 'remotes/' | sed 's/^/         /'
  fi
}

# ---------- 主循环 ----------
for name in "${REPOS[@]}"; do
  if [ "$name" = "runtime-team" ]; then
    prep_repo "$name" "$TARGET"
  else
    prep_repo "$name" "$TARGET/$name"
  fi
done

echo
echo "完成。之后日常开发:"
echo "  1. 开发前:  ./dev-prep.sh            (fetch + 切到 $DEV_BRANCH)"
echo "  2. 开发中:  在 $DEV_BRANCH 上 commit"
echo "  3. 开发完:  git push origin $DEV_BRANCH  → GitHub 上开 PR 合入 dev-1.0"
echo "  注: 本地 dev-1.0/main 等真实分支已 ff-only 同步到远端最新;"
echo "      个人分支 $DEV_BRANCH 与它们保持独立(纯本地,不绑定远端)。"
