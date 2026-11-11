#!/usr/bin/env bash
# Run on the 910C host, after uploading both source directories and installing
# the plugin editable in the user's container. No other containers are touched.
set -euo pipefail
task_root=/home/cgu135/framework-adapter-910c
container=flagos-cgu135-dev-910c
exec 9>"$task_root/.checks.lock"
flock -n 9 || { echo 'Our check suite is already running'; exit 1; }
run_id=$(date +%Y%m%d-%H%M%S)
result_dir="$task_root/results/$run_id"
mkdir -p "$result_dir"
docker inspect --format '{{.Image}} {{.State.Status}} {{json .HostConfig.Devices}}' "$container" > "$result_dir/container.txt"
docker exec "$container" pip list --format=freeze > "$result_dir/packages.txt"
docker exec "$container" timeout 120 python -m pytest \
  /workspace/vllm-plugin-FL/tests/unit_tests/dispatch -q --tb=short \
  --junitxml="/workspace/results/$run_id/unit.xml" 2>&1 | tee "$result_dir/unit.log"
# Stop on the first failure: a device failure can invalidate later results.
docker exec "$container" timeout 180 python -m pytest \
  /workspace/dev/framework-adapter/probes/test_ascend_dispatch_integration.py \
  -q -x --tb=short --junitxml="/workspace/results/$run_id/integration.xml" \
  2>&1 | tee "$result_dir/integration.log"
echo "Evidence saved to $result_dir"
