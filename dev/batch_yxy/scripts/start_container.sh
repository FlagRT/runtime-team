#!/usr/bin/env bash
# batch_yxy 单卡开发容器启动（无 docker compose v2 时的等价 docker run 方案）
# 背景：本机 docker 20.10.8 无 compose 插件，且 !override 需 compose ≥2.24；
#       等价做法参照 dev/memory/docs/goals/legacy-2.4-910c/note_新机器复现验证_910c.md
#       及 dev/framework-adapter/probes/start_locked_infer_910c.sh、dev/device-context/910C/.../start_infer_container.sh。
# 与 dev/batch_yxy/docker-compose.yml + ../compose.base.yml 等价（单卡 davinci0）。
# 用法：先 npu-smi 确认卡空闲，再 CONFIRM_DEVICE0_IDLE=yes bash dev/batch_yxy/scripts/start_container.sh
set -euo pipefail

name=flagos-batch-yxy-dev-910c
image=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
driver=/usr/local/Ascend/driver

# 单卡：davinci0 = npu-smi NPU0/chip0；其余为昇腾运行时必需管理节点
devices=(
  --device /dev/davinci0
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)

# 启动前自检（保守起见，照抄他人脚本的守门做法）
[[ $(id -un) == xianyiyuan ]] || { echo '本脚本属于 xianyiyuan'; exit 2; }
[[ ${CONFIRM_DEVICE0_IDLE:-} == yes ]] || { echo '先 npu-smi 确认卡空闲，再 CONFIRM_DEVICE0_IDLE=yes bash $0'; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "容器 $name 已存在；如需重建先 docker rm -f $name"; exit 2
fi

# 注：base 的 ../PyTorch-Plugin-FL 挂载为 B 线残留且宿主不存在，本脚本省略（避免自动建空目录）。
docker run -dit --name "$name" \
  --network host --ipc host --shm-size 512g --cap-add SYS_PTRACE --ulimit memlock=-1 \
  "${devices[@]}" \
  --mount "type=bind,src=${repo_root},dst=/workspace" \
  --mount "type=bind,src=${HOME}/.ssh/id_rsa,dst=/root/.ssh/id_rsa,readonly" \
  --mount "type=bind,src=${HOME}/.ssh/id_rsa.pub,dst=/root/.ssh/id_rsa.pub,readonly" \
  --mount "type=bind,src=${HOME}/.ssh/known_hosts,dst=/root/.ssh/known_hosts,readonly" \
  --mount "type=bind,src=${driver},dst=${driver},readonly" \
  -w /workspace \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e DO_NOT_TRACK=1 \
  -e GEMS_VENDOR=ascend \
  -e TRITON_ENABLE_TASKQUEUE=false \
  -e FLAGCX_PATH=/workspace/FlagCX/plugin/torch \
  -e HCCL_NPU_SOCKET_PORT_RANGE=16666,16676 \
  "$image" sleep infinity

docker ps --filter "name=$name" --format "{{.Names}}\t{{.Status}}\t{{.Image}}"
echo "进入容器：docker exec -it $name bash"
echo "容器内初始化：bash /workspace/dev/batch_yxy/scripts/setup_env.sh"
