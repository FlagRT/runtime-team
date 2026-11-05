#!/bin/bash
# 910C OFF 腿：vllm-ascend 锁定栈基线（10-12 复测配方）。
exec vllm serve /data_lib/models/OneRec-8B \
  --served-model-name onerec \
  --port 18320 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --max-model-len 4096
