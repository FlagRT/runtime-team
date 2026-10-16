#!/bin/bash
# 910C · T1-re 宿主侧驱动（在 npu1-27 上执行）
#   与本机纪律对齐：**带卡名额 = 同一时刻只留 1 个带卡容器** ⇒ 本容器**只挂一个 die**（+ 3 个管理设备）
#   建容器配方取自本机既有 `setup_my_container.sh`（同镜像族、同挂载风格），只把设备收窄到 1 个。
set -u

C=dc-t1re-910c-20261014
IMG=flagos-dev/pytorch-plugin-fl:manual-20260807-ascend-dev-hostnet
DIE=${DIE:-davinci4}                      # 2026-10-14 14:52 实测：NPU0/1/2/4/5/6/7 无进程；NPU3 有他人 VLLMEngineCor
WS=/mnt/raid/hliu553/dc_t1re_20261014
INNER=$WS/run_t1re_910c_20261014_in.sh

echo "########## [A] 名额清点（动手前必做）##########"
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -iE 'hliu553|dc-t1|dc-lean|flagos' || echo "  （无我方容器）"
echo "  目标 die = $DIE（挂载前确认无进程）"

echo
echo "########## [B] 建/重建容器（按**历史可用配方**逐字复用）##########"
# ⚠️ 2026-10-14 首次尝试的教训：我按"带卡名额"思路把设备**收窄成只挂 1 个 die**（`--device /dev/davinci4`）
#    ⇒ 容器内 `torch.npu.device_count() = 0`、`AssertionError: Invalid device id`、ACL 初始化失败，
#    契约不变式 rc=1、审计 rc=2、无 JSON 产物。**收窄成单 die 在本机走不通**。
#    对照本机历史**可用**容器 `flagos-infer-910c` 的 `docker inspect`：它映射的是
#      **全部 16 个 `/dev/davinci0..15` + `/dev/davinci_manager` + `/dev/devmm_svm` + `/dev/hisi_hdc`**，
#      `Ipc=private`、`Net=host`、`ShmSize=512g`、`CapAdd=[SYS_PTRACE]`、非 privileged。
#    ⇒ 设备**映射范围**与**占用名额**是两件事：名额按"被 ACL-initialized 的卡"算，
#      映射全套设备节点不等于占卡；真正用到哪张卡由运行时的 `ASCEND_RT_VISIBLE_DEVICES` 决定。
#      故此处照抄全套映射，仅把**可见卡**限定为一个空闲 die（名额纪律仍守住）。
docker rm -f "$C" >/dev/null 2>&1 || true
DEVS=""
for i in $(seq 0 15); do DEVS="$DEVS --device /dev/davinci$i"; done
DEVS="$DEVS --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"
docker run -dit \
  --name "$C" \
  --network host --shm-size 512g \
  --security-opt label=disable --cap-add CAP_SYS_PTRACE \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:rw \
  -v /mnt/raid/hliu553:/mnt/raid/hliu553 \
  $DEVS \
  "$IMG" bash >/dev/null
echo "  容器状态: $(docker inspect -f '{{.State.Status}}' $C)"

echo
echo "########## [C] 容器内自检（设备/解释器/venv）##########"
docker exec "$C" bash -lc 'ls /dev/davinci* 2>/dev/null; echo "---"; ls -l /usr/local/python3.12.13/bin/python3.12 2>/dev/null || echo "  !! 基座解释器缺失（宿主侧已移除 ⇒ 必须在容器内跑）"'
docker exec "$C" bash -lc 'ASCEND_RT_VISIBLE_DEVICES=0 /mnt/raid/hliu553/venvs/venv-infer-a/bin/python -c "
import torch, torch_npu
print(\"  torch\", torch.__version__, \"| torch_npu\", torch_npu.__version__)
print(\"  npu_count\", torch.npu.device_count(), \"| dev0\", torch.npu.get_device_name(0))
"' 2>&1 | tail -4

echo
echo "########## [D] 跑 T1-re（容器内五段）##########"
date "+[%F %T] start"
docker exec -i -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -s < "$INNER" 2>&1 | tail -50
echo "  outer rc=$?"
date "+[%F %T] end"

echo
echo "########## [E] 结果落盘与容器处置 ##########"
ls -l "$WS/out" 2>/dev/null | sed 's/^/  /'
docker stop "$C" >/dev/null 2>&1 && echo "  容器已停（保留可复用；如要彻底清理：docker rm -f $C）"
docker ps -a --format '{{.Names}}|{{.Status}}' | grep -E "$C" | sed 's/^/  /'
