#!/bin/bash
# 910C 双卡训练启动脚本（torch_fl / flagos 设备）
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
cd /workspace/dev/device-context/benchmarks
nohup /root/tf-venv-integration/bin/torchrun \
    --nproc_per_node=2 --master_port=29501 \
    train_qwen_1_5b_flagos.py > /workspace/logs/train_flagos.log 2>&1 &
echo "training started pid=$!"
