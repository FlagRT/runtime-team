#!/bin/bash
# 64GiB 锁定档 OFF 引擎（2026-10-28）—— 与 serve_lock64_on.sh 唯一差分 = 无 kv-transfer-config
# 其余逐字一致（PROFILE-PROTOCOL §7 差分唯一）。
export PYTHONPATH=/work/memory-manager:$PYTHONPATH
source /root/miniconda/bin/activate python310_torch29_cuda
exec vllm serve /work/models/OneRec-8B \
  --served-model-name onerec \
  --port 18321 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --max-model-len 4096 \
  --kv-cache-memory-bytes 68719476736
