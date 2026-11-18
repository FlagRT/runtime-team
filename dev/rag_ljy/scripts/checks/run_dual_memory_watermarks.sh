#!/usr/bin/env bash
# Existing dual-backend, dual-die RAG API only; no service restart or configuration changes.
set -euo pipefail
terminal=()
# Forward interactive Ctrl-C to Python, which stops submissions and drains accepted work.
if [[ -t 0 && -t 1 ]]; then terminal=(-it); fi
exec docker exec "${terminal[@]}" -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python -u scripts/checks/check_dual_memory_watermarks.py "$@"
