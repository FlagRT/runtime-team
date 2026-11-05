#!/bin/bash
# 910C ON 腿：同配方 + NpuSimpleCPUOffloadConnector（memory-manager 新仓路径首次真机验证）。
export PYTHONPATH=/workspace/memory-manager:$PYTHONPATH
exec vllm serve /data_lib/models/OneRec-8B \
  --served-model-name onerec \
  --port 18320 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager \
  --max-model-len 4096 \
  --kv-transfer-config '{"kv_connector":"NpuSimpleCPUOffloadConnector","kv_role":"kv_both","kv_connector_module_path":"offload.npu_offload","kv_connector_extra_config":{"cpu_bytes_to_use":8589934592}}'
