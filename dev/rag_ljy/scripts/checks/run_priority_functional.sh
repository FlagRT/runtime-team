#!/usr/bin/env bash
# Real scheduler/HTTP behavior, simulated inference. Does not restart services or use NPU.
set -euo pipefail
container="${RAG_CONTAINER:-rag-ljy-vllm-910c}"
docker exec -w /workspace/dev/rag_ljy "$container" bash -c '
set -euo pipefail
output="experiment_outputs/checks/priority_functional/$(date -u +%Y%m%dT%H%M%S.%NZ)"
mkdir -p "$output"
echo "OUTPUT=/workspace/dev/rag_ljy/$output"
echo "SCOPE=real scheduler and HTTP; simulated models/memory; no NPU acceptance" | tee "$output/scope.txt"
mkdir -p "$output/source"
cp src/rag_engine/scheduler.py src/rag_engine/server.py tests/test_priority.py tests/test_scheduler.py tests/test_server.py tests/test_generation_server.py scripts/checks/run_priority_functional.sh "$output/source/"
.venv/bin/python -m pytest -q tests/test_priority.py tests/test_scheduler.py tests/test_server.py tests/test_generation_server.py --junitxml="$output/junit.xml" 2>&1 | tee "$output/test.log"
'
