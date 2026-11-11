#!/usr/bin/env bash
# Host entry: existing API only; no container/service restart or index writes.
set -euo pipefail
exec docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python -u scripts/checks/check_generation_50.py "$@"
