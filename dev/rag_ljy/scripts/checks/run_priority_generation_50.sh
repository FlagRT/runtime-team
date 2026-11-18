#!/usr/bin/env bash
# Real NPU inference through the running RAG + OneRec API; ES by default.
# --backend milvus selects the per-request backend; no server restart.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_generation_50.sh" \
  --backend elasticsearch --priority-cycle --require-vllm-generation --parallel 4 "$@"
