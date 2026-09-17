#!/usr/bin/env bash
# Explicit host-side setup for this direction on npu1-27. Never stops other tasks.
# Run only after checking current device use and the team container-count policy.
set -euo pipefail
container=flagos-proto-infer-910c
image=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
expected_image=sha256:2e56022ae5b39930e6df12cbd69cece9156adf2a2a1136952f87c1ef0960315f
task_root=/home/cgu135/framework-adapter-910c/acceptance-20260916
model_root=/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B
[[ $(hostname) == npu1-27 ]] || { echo 'This recipe is for npu1-27 only'; exit 2; }
[[ $(id -un) == cgu135 ]] || { echo 'This recipe belongs to cgu135'; exit 2; }
[[ ${CONFIRM_DEVICE0_IDLE:-} == yes ]] || { echo 'Check npu-smi first, then set CONFIRM_DEVICE0_IDLE=yes'; exit 2; }
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$expected_image" ]] || {
  echo 'Image differs from inspected locked image; stop and review'; exit 2;
}
[[ -r "$model_root/config.json" && -r "$model_root/model.safetensors" ]] || exit 2
if docker inspect "$container" >/dev/null 2>&1; then
  echo "Container $container already exists; inspect ownership/config manually. No change made."
  exit 2
fi
# Count device-bearing or privileged running containers conservatively.
running=$(docker ps -q)
if [[ -n "$running" ]]; then
  device_containers=$(docker inspect $running --format '{{if or .HostConfig.Privileged .HostConfig.Devices}}{{.Id}}{{end}}' | awk 'NF {n++} END {print n+0}')
  [[ "$device_containers" -lt 3 ]] || { echo 'Team device-container cap reached'; exit 2; }
fi
mkdir -p "$task_root/results" "$task_root/cache"
docker run -d --name "$container" --label owner=cgu135 --label purpose=framework-adapter-20260916 \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  --cpus 8 --memory 24g --shm-size 4g --network none \
  --mount "type=bind,src=$task_root,dst=/work" \
  --mount "type=bind,src=$model_root,dst=/model,readonly" \
  --mount type=bind,src=/usr/local/Ascend/driver,dst=/usr/local/Ascend/driver,readonly \
  -e ASCEND_RT_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e DO_NOT_TRACK=1 -e OMP_NUM_THREADS=4 -e HF_HOME=/work/cache/huggingface \
  -e TRITON_CACHE_DIR=/work/cache/triton --entrypoint /bin/bash "$image" -lc 'sleep infinity'
