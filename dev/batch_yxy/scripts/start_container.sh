#!/usr/bin/env bash
# batch_yxy 单卡开发容器启动（无 docker compose v2 时的等价 docker run 方案）
# 背景：本机 docker 20.10.8 无 compose 插件，且 !override 需 compose ≥2.24；
#       等价做法参照 dev/memory/docs/goals/legacy-2.4-910c/note_新机器复现验证_910c.md
#       及 dev/framework-adapter/probes/start_locked_infer_910c.sh、dev/device-context/910C/.../start_infer_container.sh。
# 与 dev/batch_yxy/docker-compose.yml + ../compose.base.yml 等价（单卡模式）。
# 选卡：BATCH_DAVINCI_ID（默认 0）；davinci{n} = 宿主 npu-smi NPU{n/2}/chip{n%2}（n=0..15）。
#       例：davinci0 被 x-benchmark 等占用时，BATCH_DAVINCI_ID=4 换空闲卡启动。
#       容器内仍只挂这一张卡，故容器内可见设备索引恒为 0（ASCEND_RT_VISIBLE_DEVICES=0 不随卡号变）。
# 用法：先 npu-smi 确认目标卡空闲，再 CONFIRM_DEVICE0_IDLE=yes bash dev/batch_yxy/scripts/start_container.sh
set -euo pipefail

name=flagos-batch-yxy-dev-910c
image=quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
driver=/usr/local/Ascend/driver
card="${BATCH_DAVINCI_ID:-0}"

# 单卡：只挂 BATCH_DAVINCI_ID 选中的 davinci{card}；其余为昇腾运行时必需管理节点
devices=(
  --device "/dev/davinci${card}"
  --device /dev/davinci_manager
  --device /dev/devmm_svm
  --device /dev/hisi_hdc
)

# OEM 定制节点：本机 npu-smi 查 board type 依赖它（缺了报 -9005）；存在才挂，保持脚本可移植
[[ -e /dev/lqdcmi_pcidev ]] && devices+=(--device /dev/lqdcmi_pcidev)

# 启动前自检（保守起见，照抄他人脚本的守门做法）
[[ $(id -un) == xianyiyuan ]] || { echo '本脚本属于 xianyiyuan'; exit 2; }
[[ ${CONFIRM_DEVICE0_IDLE:-} == yes ]] || { echo '先 npu-smi 确认卡空闲，再 CONFIRM_DEVICE0_IDLE=yes bash $0'; exit 2; }
if docker inspect "$name" >/dev/null 2>&1; then
  echo "容器 $name 已存在；如需重建先 docker rm -f $name"; exit 2
fi

# 注：base 的 ../PyTorch-Plugin-FL 挂载为 B 线残留且宿主不存在，本脚本省略（避免自动建空目录）。
# npu-smi/dcmi/firmware/install.info 为只读挂载（镜像内没有 npu-smi；照抄本机 flagperf 容器的挂载集）。
# 已知限制：单卡容器内 npu-smi info 报 -9005（DCMI 按宿主 logic_id 全量枚举，单卡挂载对不上号）；
#           容器内验卡用 torch_npu（device_count/get_device_name），监控一律在宿主机跑 npu-smi info。
# 共享模型权重：/mnt/raid/hliu553/models 只读挂载、路径不变（含 Qwen3-Embedding-0.6B 等）。
# opencode 状态持久化：auth/API key 与会话目录落到 /workspace 挂载盘（.opencode/ 已 ignore），
#                       否则每次容器重建都要重新 auth login。
mkdir -p "${repo_root}/dev/batch_yxy/.opencode/config" "${repo_root}/dev/batch_yxy/.opencode/data"
# vLLM/vllm-ascend 源码持久化：镜像内二者是 editable 安装（真身在 /vllm-workspace/，ephemeral 层）。
# 源码已 docker cp 到 dev/batch_yxy/{vllm,vllm-ascend} 并原路挂回，改代码=改挂载盘，重建容器不丢。
docker run -dit --name "$name" \
  --network host --ipc host --shm-size 512g --cap-add SYS_PTRACE --ulimit memlock=-1 \
  "${devices[@]}" \
  --mount "type=bind,src=${repo_root},dst=/workspace" \
  --mount "type=bind,src=${HOME}/.ssh/id_rsa,dst=/root/.ssh/id_rsa,readonly" \
  --mount "type=bind,src=${HOME}/.ssh/id_rsa.pub,dst=/root/.ssh/id_rsa.pub,readonly" \
  --mount "type=bind,src=${HOME}/.ssh/known_hosts,dst=/root/.ssh/known_hosts,readonly" \
  --mount "type=bind,src=${driver},dst=${driver},readonly" \
  --mount type=bind,src=/usr/local/bin/npu-smi,dst=/usr/local/bin/npu-smi,readonly \
  --mount type=bind,src=/usr/local/dcmi,dst=/usr/local/dcmi,readonly \
  --mount type=bind,src=/usr/local/Ascend/firmware,dst=/usr/local/Ascend/firmware,readonly \
  --mount type=bind,src=/etc/ascend_install.info,dst=/etc/ascend_install.info,readonly \
  --mount type=bind,src=/mnt/raid/hliu553/models,dst=/mnt/raid/hliu553/models,readonly \
  --mount "type=bind,src=${repo_root}/dev/batch_yxy/.opencode/config,dst=/root/.config/opencode" \
  --mount "type=bind,src=${repo_root}/dev/batch_yxy/.opencode/data,dst=/root/.local/share/opencode" \
  --mount "type=bind,src=${repo_root}/dev/batch_yxy/vllm,dst=/vllm-workspace/vllm" \
  --mount "type=bind,src=${repo_root}/dev/batch_yxy/vllm-ascend,dst=/vllm-workspace/vllm-ascend" \
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
