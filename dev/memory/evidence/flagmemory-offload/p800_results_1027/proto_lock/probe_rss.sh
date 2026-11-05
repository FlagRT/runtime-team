#!/bin/bash
# 容器内 vllm 进程 RSS 探针（host RSS 起止判据用）
# EngineCore 经 setproctitle 改名（VLLM::EngineCore），须一并匹配
for p in $(pgrep -f "vllm|VLLM::|spawn_main"); do
  rss=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
  cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-70)
  title=$(cat /proc/$p/comm 2>/dev/null)
  echo "PID $p RSS_kB $rss COMM $title CMD $cmd"
done
