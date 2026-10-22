#!/usr/bin/env bash
# dryrun_merge_dev1.0.sh —— 「合入 dev-1.0 之前」的只读预检
#
# 干什么：在**临时 worktree**（`--detach`）里把 `origin/dev-1.0` 合进当前个人分支
#         （`merge --no-commit --no-ff`），打印会产生的**冲突清单 / 新增清单 / 删除清单**，
#         然后 `merge --abort` 并移除 worktree。**不改任何分支、不提交、不推送。**
#
# 为什么需要：新规范（`dev-1.0` 只接受个人分支的 merge，见 README §分支模型 与 `dev/sync_to_dev.sh`）
#   走的是**双向 merge**。个人分支若历史上有过「删除他方向目录」「夹带基座草稿」这类动作，
#   冲突面与误删风险只有**干跑**才能看清；本方向 2026-10-16 实测：27 处冲突（22 在本方向目录、
#   5 处是 modify/delete 的他方向目录）+ 1 个仅本分支存在的草稿 `dev/stack.lock.910c.yaml`。
#
# 用法（在仓根、任意分支上执行）：
#   bash dev/device-context/scripts/dryrun_merge_dev1.0.sh [dev-1.0 的 ref，默认 origin/dev-1.0]
#
# ⚠️⚠️ **陈旧 ref 陷阱（2026-10-16 实踩）**：`git fetch <remote> <branch>` **只更新 FETCH_HEAD**，
#   不动 `refs/remotes/origin/<branch>`。若直接拿 `origin/dev-1.0` 干跑，可能用的是几天前的基线，
#   **冲突面被低估**（实测：陈旧 `6af89c4` → 27 冲突；真 tip `74197bf` → **34 冲突**）。
#   先更新 ref 再跑：`git fetch origin dev-1.0:refs/remotes/origin/dev-1.0`，
#   或直接把 `FETCH_HEAD` 当参数传进来。本脚本会在两者不一致时打印告警。
#
# ⚠️ 本脚本的**第一条安全纪律**来自一次真实误操作：`git worktree add` 在「分支已被本仓检出」时
#    会失败，此时若脚本用 `cd <worktree> && git merge …` 串联，`cd` 失败后 merge 仍可能在
#    **真实工作树**上执行（2026-10-16 实踩：83 项变更、27 冲突，已 merge --abort 复原）。
#    因此本脚本：① 用 `--detach`（避免分支占用冲突）② 执行 merge 前**断言 `git rev-parse
#    --show-toplevel` == 临时 worktree**，断言不过立即退出 ③ 结束时无论如何都 `merge --abort`
#    + 移除 worktree ④ 最后打印主工作树状态供肉眼复核。
set -u

THEIRS="${1:-origin/dev-1.0}"
REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "不在 git 仓内" >&2; exit 2; }
BR="$(git symbolic-ref --short HEAD 2>/dev/null)" || { echo "当前不在分支上（detached？）" >&2; exit 2; }
WT="$(mktemp -d "${TMPDIR:-/tmp}/dryrun-merge-XXXXXX")"
OUT="${TMPDIR:-/tmp}/dryrun_merge_dev1.0.lists.txt"

cleanup() {
    if [ -d "$WT" ]; then
        ( cd "$WT" && git merge --abort >/dev/null 2>&1 ) || true
        git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true
    fi
    git -C "$REPO" worktree prune >/dev/null 2>&1 || true
    rm -rf "$WT"
}
trap cleanup EXIT

cd "$REPO" || exit 2
# 远端 ref 必须已 fetch（避免拿陈旧 FETCH_HEAD 当基线）
git rev-parse --verify -q "$THEIRS" >/dev/null || { echo "找不到 ref: $THEIRS（先 git fetch origin dev-1.0）" >&2; exit 3; }
BASE="$(git merge-base "$BR" "$THEIRS")"
echo "个人分支 : $BR ($(git rev-parse --short HEAD))"
echo "目标     : $THEIRS ($(git rev-parse --short "$THEIRS"))"
echo "共同祖先 : $BASE ($(git log -1 --format=%ci "$BASE"))"
echo "领先/落后: $(git rev-list --count "$THEIRS..$BR") / $(git rev-list --count "$BR..$THEIRS")"
# ⚠️ 陈旧 ref 告警：FETCH_HEAD 与 remote-tracking ref 不一致时提示（详见文件头）
_or="$(git rev-parse -q --verify refs/remotes/origin/dev-1.0 2>/dev/null || echo -)"
_fh="$(git rev-parse -q --verify FETCH_HEAD 2>/dev/null || echo -)"
if [ "$_or" != "-" ] && [ "$_fh" != "-" ] && [ "$_or" != "$_fh" ]; then
    echo "⚠️ 告警：origin/dev-1.0=$_or 与 FETCH_HEAD=$_fh 不一致 ⇒ 你用的 ref 可能是陈旧的。"
    echo "          先跑 git fetch origin dev-1.0:refs/remotes/origin/dev-1.0，或改用参数 FETCH_HEAD 重跑。"
fi
echo

git worktree remove --force "$WT" >/dev/null 2>&1 || true
rm -rf "$WT"
git worktree add -q --detach "$WT" "$BR" || { echo "worktree add 失败（中止）" >&2; exit 4; }

cd "$WT" || { echo "进入临时 worktree 失败（中止）" >&2; exit 5; }
# ⭐ 守卫：确认真的在临时 worktree 里，才允许动 git merge
if [ "$(git rev-parse --show-toplevel)" != "$(cd "$WT" && pwd -P)" ]; then
    echo "STOP: 当前 toplevel 不是临时 worktree —— 拒绝执行 merge（防止污染主工作树）" >&2; exit 6
fi
echo "守卫通过：toplevel=$(git rev-parse --show-toplevel)"
echo

echo "=== 干跑：git merge --no-commit --no-ff $THEIRS ==="
git merge --no-commit --no-ff "$THEIRS" > /dev/null 2>&1
echo "  merge rc=$?（1 = 有冲突，属预期；务必看下面清单）"
git status --porcelain > "$OUT"
echo

echo "=== [1] 冲突（未合并） $(git diff --name-only --diff-filter=U | wc -l | tr -d ' ') 项 ==="
git diff --name-only --diff-filter=U | sed 's/^/    /'
echo
echo "=== [2] 冲突按目录 ==="
git diff --name-only --diff-filter=U | awk -F/ '{print $1"/"$2}' | sort | uniq -c | sort -rn | sed 's/^/    /'
echo
echo "=== [3] 状态码统计（A 新增 / D 删除 / UU·AA·DU 冲突 / M 自动合并）==="
awk '{print $1}' "$OUT" | sort | uniq -c | sort -rn | sed 's/^/    /'
echo
echo "=== [4] ⚠️ 会被真删的路径（我方删除、对方未改）—— 若非空必须处置 ==="
awk '$1=="D"{print "    D " $2}' "$OUT"
echo "    （共 $(awk '$1=="D"' "$OUT" | wc -l | tr -d ' ') 项；本方向 2026-10-16 实测为 0 —— 对别人目录的删除都以 modify/delete 冲突形式出现，不会静默删掉）"
echo
echo "=== [5] modify/delete 冲突（我方删、对方改 ⇒ 需恢复为对方版本）==="
awk '$1=="DU"{print "    DU " $2}' "$OUT"
echo
echo "=== [6] 会流入本分支的他方向/共用目录（A，按顶层分组）==="
awk '$1=="A"{print $2}' "$OUT" | cut -d/ -f1-2 | sort | uniq -c | sort -rn | sed 's/^/    /' | head -14
echo
echo "完整清单已落盘：$OUT"
echo "主工作树（应只剩你自己的未跟踪改动；HEAD 不应变）："
cd "$REPO" || exit 7
git status --short | sed 's/^/    /'
echo "    HEAD=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"
