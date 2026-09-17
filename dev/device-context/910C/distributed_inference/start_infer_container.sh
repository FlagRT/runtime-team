#!/bin/bash
# 起昇腾分布式推理容器（vllm-ascend 官方镜像）
# 原则：数据/模型/venv 一律在 /mnt/raid/hliu553，不占根分区
set -e
NAME=flagos-infer-910c
IMG=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
RAID=/mnt/raid/hliu553

DEVICES=""
for i in $(seq 0 15); do DEVICES="$DEVICES --device /dev/davinci$i"; done
DEVICES="$DEVICES --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc"

docker rm -f $NAME 2>/dev/null || true

docker run -d \
  --name $NAME \
  --network host \
  --shm-size 512g \
  --cap-add SYS_PTRACE \
  $DEVICES \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v $RAID:$RAID \
  -w /workspace \
  $IMG sleep infinity

echo "container started: $NAME"
docker ps --filter name=$NAME --format "{{.Names}}\t{{.Status}}\t{{.Image}}"
