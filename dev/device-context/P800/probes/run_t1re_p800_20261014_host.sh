#!/bin/bash
# P800 · T1-re 宿主侧驱动（在 ai 侧宿主执行；P800 容器是我自己的，10-13 已同样启停过）
# 路径纪律：宿主路径由**挂载映射**推出（`/data2/hliu553 -> /workspace`），不手写
set -u

C=hliu553-device-context-p800
WP=/workspace/dc_t1re_20261014
HWP=/data2/hliu553/dc_t1re_20261014          # 由 docker inspect 的挂载推得
HERE=/data2/hliu553/dc_t1re_20261014

echo "########## [A] 容器状态与启动 ##########"
docker ps -a --filter "name=^${C}$" --format "  {{.Names}}  {{.Status}}"
docker start "$C" >/dev/null && echo "  docker start OK" || echo "  !! start 失败"
sleep 4
docker exec "$C" bash -lc 'nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader' 2>&1 | sed 's/^/  GPU /'

echo
echo "########## [B] 宿主侧工作目录与版本自证 ##########"
ls -l "$HWP/prototype/runtime/backends/base.py" 2>/dev/null || echo "  （原型尚未同步）"

echo
echo "########## [C] 容器内执行 T1-re（先离线自检，再契约不变式，再 78 项审计）##########"
date "+[%F %T] start"
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -s < "$HERE/run_t1_p800_20261014_in.sh" 2>&1 | tail -60
echo "  outer rc=$?"
date "+[%F %T] end"

echo
echo "########## [D] 结果文件落盘（宿主视角）##########"
ls -l "$HWP/out" 2>/dev/null | sed 's/^/  /'
