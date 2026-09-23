#!/usr/bin/env bash
# Task 1: race for a 2-card window, run flagcx_sync_test.py (raw broadcast/
# all_gather sync hypothesis test) in ascend-train-comm:v3, then conditionally
# the train_qwen_1_5b_npu.py sync-patched follow-up, then tear down.
set -uo pipefail

SCRATCH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COUNT_SCRIPT="$SCRATCH/count_device_containers.sh"
OUT="$SCRATCH/task1_output.log"
NAME=v3-validate-flagcx-sync-910c
IMG=flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64
RAID=/mnt/raid/hliu553

echo "waiting for device-container count <=1 (need room for our 1 container that itself holds 2 cards, cap 3 counts CONTAINERS not cards -- still only +1 container) ..." > "$OUT"
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
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$RAID/models/Qwen2.5-1.5B":/workspace/models/Qwen2.5-1.5B:ro \
  -v "$RAID/runtime-team/data":/workspace/data \
  -w /workspace \
  "$IMG" sleep infinity >> "$OUT" 2>&1

echo "docker ps AFTER launch:" >> "$OUT"
docker ps --format '{{.Names}}\t{{.Status}}' >> "$OUT" 2>&1

docker cp "$SCRATCH/flagcx_sync_test.py" "$NAME":/workspace/flagcx_sync_test.py >> "$OUT" 2>&1
docker cp /home/xliu969/runtime-team/dev/device-context/910C/distributed_training/scripts/train_qwen_1_5b_npu.py "$NAME":/workspace/train_qwen_1_5b_npu.py >> "$OUT" 2>&1
docker cp "$SCRATCH/train_qwen_1_5b_npu_syncpatch.py" "$NAME":/workspace/train_qwen_1_5b_npu_syncpatch.py >> "$OUT" 2>&1

echo "=== TASK 1 MAIN TEST: flagcx_sync_test.py (torchrun --nproc_per_node=2) ===" >> "$OUT"
docker exec \
  -e HCCL_WHITELIST_DISABLE=1 \
  "$NAME" bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && torchrun --nproc_per_node=2 --master_port=29531 /workspace/flagcx_sync_test.py" >> "$OUT" 2>&1
MAIN_RC=$?
echo "=== MAIN TEST EXIT CODE: $MAIN_RC ===" >> "$OUT"

if [ "$MAIN_RC" -eq 0 ]; then
  echo "=== sync hypothesis CONFIRMED (main test exit 0) -- running optional follow-up: train_qwen_1_5b_npu.py under sync-patch ===" >> "$OUT"
  docker exec \
    -e HCCL_WHITELIST_DISABLE=1 \
    -e MAX_STEPS=20 \
    -e BACKEND=flagcx \
    "$NAME" bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd /workspace && torchrun --nproc_per_node=2 --master_port=29532 /workspace/train_qwen_1_5b_npu_syncpatch.py" >> "$OUT" 2>&1
  FOLLOWUP_RC=$?
  echo "=== FOLLOW-UP (train_qwen_1_5b_npu.py sync-patched) EXIT CODE: $FOLLOWUP_RC ===" >> "$OUT"
else
  echo "=== sync hypothesis NOT confirmed by main test (exit $MAIN_RC) -- skipping train_qwen_1_5b_npu.py follow-up per instructions ===" >> "$OUT"
fi

echo "tearing down container" >> "$OUT"
docker rm -f "$NAME" >> "$OUT" 2>&1

echo "docker ps AFTER cleanup:" >> "$OUT"
docker ps --format '{{.Names}}\t{{.Status}}' >> "$OUT" 2>&1

echo "TASK1 DONE main_rc=$MAIN_RC" >> "$OUT"
