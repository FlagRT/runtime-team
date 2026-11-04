#!/usr/bin/env bash
# Run on the host. Each trial has a separate cold-start setup, never an in-trial restart.
set -euo pipefail
if [[ "${1:-}" != "--run" ]]; then
  if [[ $# -ne 0 ]]; then echo 'Usage: bash scripts/checks/run_measured_acceptance.sh [--run]' >&2; exit 2; fi
  docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
    .venv/bin/python scripts/runtime/restart_memory_api.py
  exit 0
fi
for mode in burst staggered continuous; do
  docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
    .venv/bin/python scripts/runtime/restart_memory_api.py --run
  docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
    .venv/bin/python -u scripts/checks/check_measured_watermarks.py --run \
    --modes "$mode" --count 2 --warmup-profile 1024:32 --profiles 2048:32
done
docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python -u scripts/checks/check_stale_memory_live.py --run
