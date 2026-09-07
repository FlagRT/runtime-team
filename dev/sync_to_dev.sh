#!/usr/bin/env bash
# 标准提交流程：将当前个人开发分支的变更同步进 dev-1.0 并推送
#
# 规范（自建仓 runtime-team，见 README「分支模型 · 规范（自建仓）」）：
#   - 直接 commit / rebase 只发生在个人开发分支
#   - dev-1.0 只接受个人分支的 merge（本地 merge 后 push 即可，无需 PR）
#   - 共享分支（dev-1.0 / main）只用 merge，不用 rebase
#
# 用法：在【个人开发分支】上、工作区干净时执行
#   bash dev/sync_to_dev.sh
#
# 任一步失败（非快进 / 合并冲突 / 远端分叉）即中止，按提示手动处理后重跑。

set -euo pipefail

DEV_BRANCH="dev-1.0"
REMOTE="origin"

cd "$(git rev-parse --show-toplevel)"

FEATURE_BRANCH="$(git symbolic-ref --short HEAD)"

if [ "$FEATURE_BRANCH" = "$DEV_BRANCH" ] || [ "$FEATURE_BRANCH" = "main" ]; then
  echo "STOP: 当前在共享分支 '$FEATURE_BRANCH'，请切到个人开发分支再执行" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "STOP: 工作区有未提交变更，请先 commit 或 stash" >&2
  exit 1
fi

echo "==> 个人分支: $FEATURE_BRANCH   目标: $REMOTE/$DEV_BRANCH"

# 1. 远端 dev 同步到本地 dev-1.0（必须快进，不产生新提交）
git fetch "$REMOTE"
git checkout "$DEV_BRANCH"
git merge --ff-only "$REMOTE/$DEV_BRANCH"

# 2. dev-1.0 merge 进个人分支（冲突在此解决，解决后重跑本脚本）
git checkout "$FEATURE_BRANCH"
git merge --no-ff "$DEV_BRANCH" -m "Merge $DEV_BRANCH into $FEATURE_BRANCH"

# 3. 处理好的个人分支合入本地 dev-1.0
git checkout "$DEV_BRANCH"
git merge --no-ff "$FEATURE_BRANCH" -m "Merge $FEATURE_BRANCH into $DEV_BRANCH"

# 4. 校验远端无分叉（origin/dev-1.0 是本地 dev-1.0 的祖先）后推送
git fetch "$REMOTE"
if git merge-base --is-ancestor "$REMOTE/$DEV_BRANCH" "$DEV_BRANCH"; then
  echo "OK 无分叉"
else
  echo "STOP: $REMOTE/$DEV_BRANCH 已分叉，回到步骤 1 重新同步" >&2
  git checkout "$FEATURE_BRANCH"
  exit 1
fi
git push "$REMOTE" "$DEV_BRANCH"

# 5. 回个人分支
git checkout "$FEATURE_BRANCH"
echo "==> 完成：$FEATURE_BRANCH 已同步进 $REMOTE/$DEV_BRANCH"
