#!/bin/bash
# 引擎切换 ON→OFF runbook（2026-10-28）——唯一一次协议必需的 serve 重启（PLAN §1 引擎史重置，如实记录）
# 分步执行，每步短命令（P800 SSH 纪律）。本脚本由执行员逐步手动调用，不整跑。
# 用法: switch_engine.sh <step>
#   check_done   - 确认 ON 链完成
#   kill_on      - 停 ON serve（SIGTERM pid 167，等待进程树退出）
#   verify_free  - 真值验证卡已释放（mem_get_info）
#   archive_mk   - 归档 ON 腿阶段 /tmp markers → proto_lock/ON/markers/，/tmp 改名留痕
#   start_off    - 启动 OFF serve（重定向在 exec 外层）
#   wait_ready   - 轮询 OFF 就绪 + 池行
set -u
D=/data1/chenyuanxin/p800-1027
STEP=${1:-check_done}
case $STEP in
  check_done)
    cat $D/proto_lock/ON/chain_status.json
    ;;
  kill_on)
    docker exec mem-profile-p800 bash -c 'kill 167 2>/dev/null; sleep 1; ps -eo pid,comm | grep -E "vllm|VLLM" | head'
    ;;
  verify_free)
    sleep 5
    docker exec mem-profile-p800 bash -c 'ps -eo pid,comm | grep -E "vllm|VLLM" | grep -v grep | head -3; source /root/miniconda/bin/activate python310_torch29_cuda && python /work/p800-1027/probe_meminfo.py 2>/dev/null | tail -1'
    ;;
  archive_mk)
    docker exec mem-profile-p800 bash -c "mkdir -p /work/p800-1027/proto_lock/ON/markers && cp /tmp/w2x_marker_*.txt /work/p800-1027/proto_lock/ON/markers/ 2>/dev/null; for f in /tmp/w2x_marker_transfers.txt /tmp/w2x_marker_diag.txt /tmp/w2x_marker_register.txt /tmp/w2x_marker_swap.txt; do [ -f \$f ] && mv \$f \${f}.ON_archived; done; ls /work/p800-1027/proto_lock/ON/markers/ 2>/dev/null; ls /tmp/ | grep w2x"
    ;;
  start_off)
    docker exec -d mem-profile-p800 bash -c 'bash /work/p800-1027/proto_lock/serve_lock64_off.sh > /work/p800-1027/logs/serve_lock64_off_20261028.log 2>&1 < /dev/null'
    sleep 3
    docker exec mem-profile-p800 bash -c 'ps -eo pid,cmd | grep "vllm serve" | grep -v grep | head -2 | cut -c1-150'
    ;;
  wait_ready)
    L=/work/p800-1027/logs/serve_lock64_off_20261028.log
    docker exec mem-profile-p800 bash -c "grep -E 'GPU KV cache size|Maximum concurrency|Application startup complete|ERROR|Error' $L | tail -5; echo ---; tail -2 $L | cut -c1-160"
    ;;
  neighbors)
    xpu-smi 2>/dev/null | grep -E '^\|   [0-9] ' | awk '{print $2, $10, $11}' | head -10
    ;;
esac
