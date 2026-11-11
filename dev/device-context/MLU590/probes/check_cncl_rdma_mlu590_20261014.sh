#!/bin/bash
# MLU590 · 「CNCL RDMA 通路」能力核查 —— 审计 A4 的**归因复核**
# 在 Mlu-1 宿主上执行：bash -s < 本脚本
#
# 【要回答的问题】
#   既有文档写「CNCL 的 RDMA 路径未验证 ⇒ **非本层缺口**（厂商/网络层）」。
#   「未验证」是事实，但**归因**从未实测过。而 CNCL 自己的告警给的是**可操作指引**：
#     "Failed to open libibverbs.so[.1]. If you want to use RDMA transport,
#      you need add libibverbs.so path to LD_LIBRARY_PATH."
#   ⇒ 若库里其实存在、只是没进路径，那本条的定性应改为「**我方容器未开通**（可试）」，
#      而不是「厂商/网络层不足」。本脚本就是判这一条。
#
# 【判据】
#   (a) 宿主是否有 RDMA 网卡（lspci / /sys/class/infiniband / /dev/infiniband）
#   (b) 机器上是否存在 libibverbs.so* / libmlx5.so*
#   (c) 容器内的 LD_LIBRARY_PATH / ldconfig 能否找到它们
#   (d) `libcncl.so*` 是否**自身就带 RDMA 传输路径**（引用 mlx5/ibverbs 符号、含 RDMA 相关环境变量名）
#   (e) 版本是否相容：可得的 libmlx5 里是否有 CNCL 需要的 `mlx5dv_modify_qp_udp_sport`
set -u

C=${C:-dc-mlu590-hliu553}
C2=${C2:-dc-mlu590-vllm-hliu553}

echo "===== [0] 时间戳 / 主机 ====="
date "+[%F %T]"; hostname; . /etc/os-release 2>/dev/null && echo "  $PRETTY_NAME"

echo
echo "===== [1] 硬件：RDMA 网卡与设备节点 ====="
printf "  lspci 命中(mlx/infiniband) 数="; lspci 2>/dev/null | grep -icE "mellanox|infiniband"
lspci 2>/dev/null | grep -iE "mellanox|infiniband" | head -3 | sed 's/^/    /'
echo "  /sys/class/infiniband:"; ls /sys/class/infiniband 2>&1 | head -5 | sed 's/^/    /'
echo "  /dev/infiniband:"; ls /dev/infiniband 2>&1 | head -5 | sed 's/^/    /'
echo "  ibv_devinfo/ibstat 是否可用:"; command -v ibv_devinfo ibstat 2>/dev/null | sed 's/^/    /' || echo "    (都没有)"

echo
echo "===== [2] 库：宿主上是否存在 libibverbs / libmlx5 ====="
for so in libibverbs.so libibverbs.so.1 libmlx5.so libmlx5.so.1; do
    printf "    %-18s " "$so"
    _h=$(find /usr /opt /usr/local /lib /srv -maxdepth 6 -name "$so" 2>/dev/null | head -3 | tr '\n' ' ')
    echo "${_h:-（未找到）}"
done
echo "  包管理器痕迹:"; { rpm -qa 2>/dev/null | grep -iE "rdma|ibverbs|libmlx" ; dpkg -l 2>/dev/null | grep -iE "rdma|ibverbs|libmlx" | awk '{print $2}' ; } | head -6 | sed 's/^/    /'

echo
echo "===== [3] 容器内视角（LD_LIBRARY_PATH / ldconfig / 设备） ====="
for cc in "$C" "$C2"; do
    echo "  --- 容器 $cc ---"
    docker ps --format '{{.Names}}' | grep -q "^$cc$" || { echo "    （容器不在运行，跳过）"; continue; }
    docker exec "$cc" bash -lc 'echo "    LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-（空）}"
printf "    ldconfig 命中(ibverbs|mlx5)="; ldconfig -p 2>/dev/null | grep -cE "libibverbs|libmlx5"
ls /dev/infiniband 2>&1 | head -3 | sed "s/^/    dev_infiniband: /"
for so in libibverbs.so.1 libmlx5.so.1; do printf "    %-18s " "$so"; find /usr /opt -maxdepth 6 -name "$so" 2>/dev/null | head -2 | tr "\n" " "; echo; done'
done

echo
echo "===== [4] libcncl 是否自带 RDMA 传输路径（符号 + 环境变量名）====="
CNCL=$(docker exec "$C" bash -lc 'find / -maxdepth 7 -name "libcncl.so*" 2>/dev/null | head -3' 2>/dev/null | tr '\n' ' ')
echo "  libcncl 路径：${CNCL:-（未找到）}"
for lib in $CNCL; do
    echo "  --- $lib ---"
    docker exec "$C" bash -lc "readelf -d '$lib' 2>/dev/null | grep -iE 'NEEDED' | head -12" | sed 's/^/    /'
    printf "    未解析的 ibverbs/mlx5 符号: "
    docker exec "$C" bash -lc "nm -D --undefined-only '$lib' 2>/dev/null | grep -icE 'ibv_|mlx5_'" 2>/dev/null || echo "?"
    echo "    CNCL 环境变量名（含 RDMA/IB/ROCE/TCP 关键词）:"
    docker exec "$C" bash -lc "strings '$lib' 2>/dev/null | grep -E '^CNCL_[A-Z0-9_]+' | sort -u" | head -40 | sed 's/^/      /'
    break
done

echo
echo "===== [5] 版本相容性：可得 libmlx5 是否有 CNCL 需要的符号 ====="
MLX=$(docker exec "$C" bash -lc 'find /usr /opt -maxdepth 6 -name "libmlx5.so*" 2>/dev/null | head -2' 2>/dev/null | tr '\n' ' ')
echo "  容器内 libmlx5：${MLX:-（未找到）}"
for m in $MLX; do
    printf "    %s : mlx5dv_modify_qp_udp_sport => " "$m"
    docker exec "$C" bash -lc "nm -D '$m' 2>/dev/null | grep -c 'mlx5dv_modify_qp_udp_sport'" 2>/dev/null || echo "?"
done

echo
echo "===== [6] 复述既有告警原文（用于比对是否同源） ====="
grep -ah -m4 -E "libibverbs|libmlx5|MLU_LINK topo" /srv/hliu553/dc_t7_20261014/tp2_eager1/vllm_serve.log 2>/dev/null | sed 's/^/  /' \
    || grep -ah -m4 -E "libibverbs|libmlx5|MLU_LINK topo" /srv/hliu553/dc_mlu_regen_20261008/*/*.log 2>/dev/null | sed 's/^/  /' \
    || echo "  （未取到，需在训练腿日志里取）"
echo
echo "===== 结束 $(date '+%F %T') ====="
