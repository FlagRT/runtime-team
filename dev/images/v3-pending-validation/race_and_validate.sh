#!/usr/bin/env bash
# Poll tightly for device-container count <=2 (leaving room for our 1 within
# the machine-wide cap of 3), then IMMEDIATELY launch a single 1-card
# validation container, run the Step 5 script, capture output, and tear down.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COUNT_SCRIPT="$HERE/count_device_containers.sh"
VALIDATE_PY="$HERE/v3_step5_validate.py"
OUT="$HERE/v3_step5_output.log"
NAME=v3-validate-train-910c-r2
IMG=flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64
RAID=/mnt/raid/hliu553

echo "waiting for device-container count <=2 ..." > "$OUT"
for i in $(seq 1 240); do
  n=$("$COUNT_SCRIPT")
  echo "$(date '+%H:%M:%S') device-containers=$n" >> "$OUT"
  if [ "$n" -le 2 ]; then
    echo "CAUGHT WINDOW at n=$n, launching immediately" >> "$OUT"
    break
  fi
  sleep 3
done

echo "docker ps BEFORE launch:" >> "$OUT"
docker ps --format '{{.Names}}\t{{.Status}}' >> "$OUT" 2>&1

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$NAME" \
  --network host \
  --shm-size 64g \
  --cap-add SYS_PTRACE \
  --device /dev/davinci0 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$RAID":"$RAID":ro \
  -w /workspace \
  "$IMG" sleep infinity >> "$OUT" 2>&1

echo "docker ps AFTER launch:" >> "$OUT"
docker ps --format '{{.Names}}\t{{.Status}}' >> "$OUT" 2>&1

docker cp "$VALIDATE_PY" "$NAME":/tmp/v3_step5_validate.py >> "$OUT" 2>&1

echo "=== RUNNING VALIDATION SCRIPT ===" >> "$OUT"
docker exec \
  -e SMOKE_MODEL_PATH=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
  "$NAME" \
  /usr/local/python3.11.15/bin/python3 /tmp/v3_step5_validate.py >> "$OUT" 2>&1
RC=$?
echo "=== VALIDATION SCRIPT EXIT CODE: $RC ===" >> "$OUT"

echo "tearing down container" >> "$OUT"
docker rm -f "$NAME" >> "$OUT" 2>&1

echo "docker ps AFTER cleanup:" >> "$OUT"
docker ps --format '{{.Names}}\t{{.Status}}' >> "$OUT" 2>&1

echo "DONE rc=$RC" >> "$OUT"
exit "$RC"
