# 运行指南

更新：2026-10-14。默认选择 vLLM embedding，启动顺序是数据库 → embedding
引擎 → 查询 API。以下命令均从指定目录执行。

## 宿主机：三个容器

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
docker compose --env-file .env -f docker_compose/compose.milvus.yml up -d --wait
docker compose --env-file .env -f docker_compose/compose.elasticsearch.yml up -d --wait
docker start rag-ljy-vllm-910c
docker exec -it rag-ljy-vllm-910c bash
```

现有 NPU 容器挂载整个仓库为 `/workspace`，只映射物理 `/dev/davinci7` 和
必要控制设备。容器内使用逻辑 `npu:0`。启动容器不会自动启动两个 Python 服务。
已有同端口服务时先查看 `/health`，避免重复启动。

当前Elasticsearch宿主机端口为 `127.0.0.1:9201`（`.env` 的 `ELASTICSEARCH_PORT=9201`），
映射到容器内9200；RAG连接地址为 `http://127.0.0.1:9201`。2026-10-14因宿主机
9200被其他监听服务占用而调整。容器内部healthcheck仍查询9200。

新建 NPU 容器（需要镜像、设备与模型挂载就绪）：

```bash
docker compose --env-file .env -f ../compose.base.yml   -f docker_compose/compose.vllm.yml up -d runtime-dev
```

该文件与根目录 `docker-compose.yml` 是替代入口，仅选择一个。
重建 NPU 容器可能需要重新安装本地 `.venv`。

## 容器内：Python 与设备

```bash
cd /workspace/dev/rag_ljy
source .venv/bin/activate
export PYTHONPATH="/workspace/dev/device-context/prototype${PYTHONPATH:+:$PYTHONPATH}"
export DC_BACKEND=ascend
export RAG_DEVICE=npu:0
python scripts/checks/check_device.py --device "$RAG_DEVICE"
```

首次安装/修复虚拟环境：`bash scripts/runtime/setup_device_env.sh`。
脚本保留镜像中的 torch、torch_npu、vLLM 依赖版本。

## 容器内：两个常驻进程

终端1：

```bash
cd /workspace/dev/rag_ljy
source .venv/bin/activate
python scripts/runtime/serve_embeddings.py --device-index 0 --port 8089
```

启动器设置 `TRITON_ENABLE_TASKQUEUE=true`，补充 CANN ACL 路径，使用本地
Qwen3-Embedding-0.6B、BF16、LAST pooling。等待
`curl -f http://127.0.0.1:8089/health` 成功。

终端2：

```bash
cd /workspace/dev/rag_ljy
source .venv/bin/activate
export DC_BACKEND=ascend
export RETRIEVAL_BACKEND=milvus
export MILVUS_COLLECTION=nfcorpus_chunks_v1
export ELASTICSEARCH_INDEX=nfcorpus-chunks-v1
python scripts/runtime/serve_queries.py --device npu:0 --port 8088   --max-inflight 8 --retrieval-workers 4 --embedding-workers 2 --stage-queue-capacity 8
```

`.env` 与代码默认选择 `RAG_EMBEDDING_BACKEND=vllm`。引擎不可用时报错。
精排通过 device-context 在 `npu:0` 使用 Transformers；embedding 的设备由
独立启动器控制。切换数据库时设 `RETRIEVAL_BACKEND=elasticsearch` 并重启 API。

```bash
curl -f http://127.0.0.1:8088/health
curl -f -H 'Content-Type: application/json'   -d '{"query":"How does nutrition affect health?"}' http://127.0.0.1:8088/query
```

`--embedding-workers` 控制vLLM提交并发，`--stage-queue-capacity` 控制每阶段等待槽。
全局请求容量满返回503，已接收请求遇阶段满则等待空间。

`/health` 应显示 `embedding_backend: vllm`；`device: remote-vllm` 描述 embedding
适配器，精排仍在上述 NPU。

查询API默认通过device-context每秒采集一次设备内存，缓存结果位于
`/health` 的 `device_memory` 字段，包括 `status`、`stats`、`sampled_at`、
`age_seconds` 和 `stale`。可设置 `--memory-probe-interval 2` 调整间隔，
或用 `--no-memory-probe` 关闭。当前仅采样，尚不根据内存自动拒绝或扩大并发。
统计范围与兼容回退的说明见 [device-context接入](device-context.md)。

只检查设备内存，无需加载模型或启动数据库：

```bash
python scripts/checks/check_memory.py --device npu:0 --count 5
```

结果保存为 `artifacts/checks/memory_probe/<日期时间>-<ID>/report.md`，每轮只有一个文件。

## 同语料双库入库

宿主机数据：`/mnt/raid/jliu171/data/nfcorpus/processed/nfcorpus_documents.jsonl`。
复制到 NPU 容器后，在两个数据库及 embedding 引擎就绪时执行：

```bash
# 宿主机
docker cp /mnt/raid/jliu171/data/nfcorpus/processed/nfcorpus_documents.jsonl   rag-ljy-vllm-910c:/tmp/nfcorpus_documents.jsonl
# 容器内
python scripts/data/ingest_corpus_both.py --input /tmp/nfcorpus_documents.jsonl   --device npu:0 --batch-size 8 --write-batch-size 64
python scripts/checks/verify_rebuilt_stores.py   --output /tmp/rag-store-verification.json
```

默认分块为1200字符、重叠150字符，共6,543块。每批向量复用于两个数据库。
入库脚本创建不存在的资源、按固定 ID 写入，最后核对计数并检查两路搜索。
两个数据库的写入并非同一个事务；失败后使用相同输入和参数重新执行。

## 持久化与停止

当前 `.env` 指向：

- Milvus：`/mnt/raid/jliu171/milvus-data-vllm-20261009`，UID/GID `999:999`，权限750。
- Elasticsearch：`/mnt/raid/jliu171/elasticsearch-data-vllm-20261009`，UID/GID `1000:0`，权限750。

Milvus 内部运行 etcd；无需单独的 `rag-ljy-milvus-etcd` 容器。内嵌配置在
[compose.milvus.yml](../docker_compose/compose.milvus.yml)。外置部署保留在
[compose.milvus.external-etcd.yml](../docker_compose/compose.milvus.external-etcd.yml)，
切回旧库时须停止内嵌部署，并将数据目录改回旧路径。两种部署不可同时读写同一个目录。
旧数据目录仍保留；[重建证据](../artifacts/rebuilds/vllm-rebuild-2026-10-09/manifest.json)
记录计数和重启验证。

```bash
# 宿主机：停止当前三个容器
docker stop rag-ljy-vllm-910c rag-ljy-milvus rag-ljy-elasticsearch
```

历史调试进程日志可能位于容器 `/tmp`，正式保存的结果见
[artifacts/README.md](../artifacts/README.md)。
