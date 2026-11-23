# 从零到判定：逐步操作

> 正常使用请直接跑 [`run.sh`](run.sh)（`XPU_CARD=<n> MODEL_DIR=/path bash run.sh`），
> 它已经把下面这几步自动化了。本文件是给要手动排查、或者想看 run.sh 内部到底做了什么的人看的。

1. 起容器（或复用现有常驻容器 `flagrt-kunlun-infer-v1-env-v1-p800a`）：按现行
   `kunlun-infer-vllm/v1` 的 `env.baseline.yaml` 核对镜像 id 后 `docker compose up -d`。
2. 进容器前先在当前 shell 里 `export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
   VLLM_HOST_IP=127.0.0.1`（容器 `--network none`，否则 vllm 分布式初始化会去找 `eth0`）。
3. `docker exec` 进容器，确认模型目录存在：`/home/models/Qwen3-Embedding-0.6B/`（没有就从
   hf-mirror.com 下载，sha256 见 README 前置条件表，下载后留着方便下次复用）。
4. 跑单入口命令（见 README「复现」节），确认先拿到 `overall: FAIL` 的对照组结果——**跳过这步
   直接打补丁会分不清是这次没打对、还是环境本来就没问题**。
5. 打补丁：`docker exec <容器> sed -i '849s/.../.../' /workspace/vllm-plugin-FL/vllm_fl/worker/worker.py`
   （行号以实际镜像内文件为准，先 `grep -n` 确认）。不需要重建镜像、不需要重启容器。
6. 重跑同一条单入口命令，确认 `overall: PASS`。
7. 收尾：这只是验证用的临时容器内文件改动，不用恢复（反正不影响镜像本体，容器重建后会还原）；
   不碰其他用户的容器；用完 `docker compose down` 或保留常驻容器均可，视后续是否还要用。
