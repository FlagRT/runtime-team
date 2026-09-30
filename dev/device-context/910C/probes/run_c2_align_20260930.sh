#!/bin/bash
# C2：把 910C 宿主副本对齐到远端 tip（不删除任何东西：冲突的未跟踪文件 mv 到备份目录）
# 反复「试 ff → 取冲突清单 → 移动到备份」，最多 6 轮。
set -u
cd /mnt/raid/hliu553/runtime-team
B=/mnt/raid/hliu553/host_copy_untracked_backup_20260930
mkdir -p "$B"
LOG=/mnt/raid/hliu553/c2_align.log
: > "$LOG"

for i in 1 2 3 4 5 6; do
  echo "=========== 轮 $i ===========" >> "$LOG"
  rm -f /tmp/c2err_$i
  timeout 200 git merge --ff-only FETCH_HEAD > /tmp/c2out_$i 2> /tmp/c2err_$i
  rc=$?
  echo "merge rc=$rc" >> "$LOG"
  if [ "$rc" = "0" ]; then
    echo "ALIGN_OK" >> "$LOG"
    break
  fi
  # 取 "would be overwritten by merge" 段落里的文件清单
  awk '/would be overwritten by merge/{f=1;next} f&&/^[[:space:]]*$/{f=0} f' /tmp/c2err_$i \
    | sed 's/^[[:space:]]*//' | grep -E '^dev/device-context/' | sort -u > /tmp/c2list_$i
  cnt=$(wc -l < /tmp/c2list_$i)
  echo "冲突文件 $cnt 个" >> "$LOG"
  if [ "$cnt" = "0" ]; then
    echo "无冲突可移动但 ff 仍失败 ⇒ 停止（需人工看 /tmp/c2err_$i）" >> "$LOG"
    break
  fi
  m=0
  while read -r f; do
    [ -z "$f" ] && continue
    mkdir -p "$B/$(dirname "$f")"
    mv -n "$f" "$B/$f" 2>>"$LOG" && m=$((m+1))
  done < /tmp/c2list_$i
  echo "已移动 $m 个到备份" >> "$LOG"
done

echo "===== 结果 =====" >> "$LOG"
git log --oneline -1 >> "$LOG"
echo "HEAD=$(git rev-parse HEAD)" >> "$LOG"
echo "FETCH_HEAD=$(git rev-parse FETCH_HEAD)" >> "$LOG"
echo "剩余脏文件数=$(git status --porcelain 2>/dev/null | wc -l)" >> "$LOG"
echo "备份文件数=$(find "$B" -type f 2>/dev/null | wc -l)" >> "$LOG"
echo "DONE" >> "$LOG"
