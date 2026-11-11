#!/bin/bash
# G4 64GiB 锁定档部署（2026-10-27）—— PLAN_规范验证跑_真模型 §2 固定表
# OneRec-8B, TP=1, gmu 0.8, max-model-len 4096, enforce-eager,
# kv-cache-memory-bytes=64GiB(68719476736), Host 池 16GiB(17179869184),
# connector=XpuSimpleCPUOffloadConnector(offload.xpu_offload, 新仓 /work/memory-manager)。
# 先例锚点：2026-10-25 同配置池=466,032 tok。
export PYTHONPATH=/work/memory-manager:$PYTHONPATH
source /root/miniconda/bin/activate python310_torch29_cuda
exec vllm serve /work/models/OneRec-8B \
  --served-model-name onerec \
  --port 18321 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --max-model-len 4096 \
  --kv-cache-memory-bytes 68719476736 \
  --kv-transfer-config '{"kv_connector":"XpuSimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_module_path":"offload.xpu_offload","kv_connector_extra_config":{"cpu_bytes_to_use":17179869184}}'
