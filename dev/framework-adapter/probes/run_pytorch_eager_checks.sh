#!/usr/bin/env bash
# Host-side runner. Reuses only the user's existing single-device container.
set -euo pipefail
task_root=/home/cgu135/framework-adapter-910c
container=flagos-cgu135-dev-910c
exec 9>"$task_root/.checks.lock"
flock -n 9 || { echo 'Our check suite is already running'; exit 1; }
run_id="pytorch-$(date +%Y%m%d-%H%M%S)"
result_dir="$task_root/results/$run_id"
mkdir -p "$result_dir"
docker inspect --format '{{.Image}} {{.State.Status}}' "$container" > "$result_dir/container.txt"
docker exec -e PYTHONPATH=/workspace/dev/framework-adapter/probes "$container" \
  timeout 180 python -m pytest /workspace/dev/framework-adapter/probes/test_pytorch_eager_adapter.py \
  -q -x --tb=short --junitxml="/workspace/results/$run_id/tests.xml" \
  2>&1 | tee "$result_dir/tests.log"
docker exec -e PYTHONPATH=/workspace/dev/framework-adapter/probes "$container" \
  timeout 120 python /workspace/dev/framework-adapter/probes/pytorch_eager_demo.py \
  2>&1 | tee "$result_dir/demo.log"
echo "Evidence saved to $result_dir"
