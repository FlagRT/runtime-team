# 运行指南

更新：2026-11-11。默认选择 vLLM embedding，启动顺序是数据库 → embedding
引擎 → 查询 API。以下命令均从指定目录执行。

当前新增OneRec生成模式与物理卡0/die0部署，见[OneRec接入](generation.md)。
下面基础启动示例保留检索模式；生成模式的完整配置及当前已验证范围以该文档和根目录[status.md](../status.md)为准。

## 宿主机：三个容器

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
docker compose --env-file .env -f docker_compose/compose.milvus.yml up -d --wait
docker compose --env-file .env -f docker_compose/compose.elasticsearch.yml up -d --wait
docker start rag-ljy-vllm-910c
docker exec -it rag-ljy-vllm-910c bash
```

基础NPU Compose映射`/dev/davinci1`；2026-11-11实际主容器使用OneRec overlay，
选择`/dev/davinci0`（物理卡0/chip0），容器内仍为`npu:0`。
上述`docker start`只启动已有容器，不会应用新的Compose映射。
仓库挂载为 `/workspace`。启动容器不会自动启动两个 Python 服务。
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
或用 `--no-memory-probe` 关闭。默认只采样；内存准入需按下面的配置显式开启，
且不能同时关闭采样。`--max-inflight 0` 可在显存准入开启时取消数量上限；阶段等待队列仍有界，已接收工作等待队列空间。2026-10-29当前API已使用此配置。
统计范围与兼容回退的说明见 [device-context接入](device-context.md)。

只检查设备内存，无需加载模型或启动数据库：

```bash
python scripts/checks/check_memory.py --device npu:0 --count 5
```

结果保存为 `artifacts/checks/memory_probe/<日期时间>-<ID>/report.md`，每轮只有一个文件。

## 可选：按设备内存接收请求

在上面的终端2环境中，用下列命令替换原查询API启动命令；需要重新启动API进程。
当前部署使用动态估算：每条请求按实际token、有效batch、模型结构和执行配置分别估算embedding与rerank新增峰值，不再统一预留2GiB。

```bash
python scripts/runtime/serve_queries.py --device npu:0 --port 8088 \
  --max-inflight 0 --memory-admission --memory-budget-mode dynamic \
  --memory-probe-interval 0.1 --embedding-workers 8 --max-reranker-batch-size 32 \
  --memory-allow-unverified-scope
```

默认高水位为0.85（达到即暂停）、低水位为0.75、安全余量5120MiB、读数最大年龄3秒；分别用
`--memory-high-watermark`、`--memory-low-watermark`、`--memory-safety-mib`、
`--memory-max-sample-age`调整。水位不是动态学习值；每次有效采样均检查设备实测占用，达到85%暂停，降至75%及以下恢复。未暂停时，新请求须满足
`实测空闲 − 已有预留 − 新请求预算 ≥ 安全余量`，不再用高水位限制加入新请求后的总量。
预算只参与新请求的余量检查；85%/75%滞回只使用device-context实测占用。实测占用与预算相加的effective_used_percent仅供诊断，不控制压力暂停。

`POST /memory/estimate`接受与`/query`相同的JSON，只在CPU上返回估算及其组成，不预留预算、不运行模型。真正的`/query`会重新估算并原子地检查/登记预算。成功响应头`X-Memory-Estimate-MiB`及服务日志记录该请求金额；拒绝响应区分`request_budget`、`memory_pressure`、`stale_telemetry`等原因。

当前解析估算仍标为`analytical_uncalibrated`，不等于已验证的通用上界。精排forward前读取最新显存并按实际padding形状复核，必要时缩小实际batch；分配器峰值观测超过估算时停止新准入并报告fault。当前部署文档token保护上限640，实时核对6,543块最长578，均在此范围内。未知远程embedding配置、dtype/图模式未明确等情况会要求验证，不套用默认2GiB。

旧`--memory-request-mib`仅能配合显式`--memory-budget-mode fixed`用于历史兼容，不是当前实验配置。最终验收目标见[要求清单](memory-admission-requirements.md)；两库未含生成的85%三模式验收已通过，新增生成后的专项未执行，见根目录[status.md](../status.md)。

`--memory-allow-unverified-scope` 是明确允许使用当前runtime未标明统计范围的
读数：2026-10-14在物理NPU7观察到跨进程占用，但这不保证所有后端或回退路径
都有整卡视角。其他环境应先验证。没有该选项时，只接受 `scope: device`，
当前未验证范围的读数会导致503；不会默默按“内存充足”处理。

余量不足返回503、`Retry-After: 1`和 `X-Request-ID`，响应detail包含原因；
超过高水位后要回落到低水位或以下才解除暂停；解除后仍须通过空闲内存预算检查。未就绪、过期、失败或不完整的采样也会
暂停新接收。已接收请求继续完成，预算保留到实际处理结束，并等结束后开始的
一次有效采样再释放，避免复用旧读数。预算是逻辑记账，不是实际预分配。

`GET /health` 的 `memory_admission` 展示 `ready` / `paused` / `unavailable`、
当前预算、待释放预算、配置及最后一次决策；顶层 `ready` 仅表示模型已加载。
启动后先等待 `memory_admission.status` 为 `ready` 再开始正常查询测试。
服务日志的 `event=memory_admission` 保存每次内存准入决策，不记录查询原文。
当前只保护被探测的那张设备，独立vLLM进程应使用同一张设备；尚未接入主机内存
限额或OOM恢复，也无法阻止其他进程突然抢占设备内存。

2026-10-20受控测试与169项完整回归通过，见
[单文件准入报告](../artifacts/checks/memory_admission/2026-10-20/report.md)。
这次未做真实NPU内存压力测试；原有200次双库结果不代表启用此策略后的复测结果。

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

### 当前85%验证环境的额外运行配置

`RAG_PROFILE_FIRST_FORWARD=1`用于核实真实融合attention，`RAG_RERANK_DOCUMENT_TOKENS=640`与当前语料长度核对一致。`PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.70`及allocator fraction 0.99控制正常缓存回收，不预分配显存。动态估算使用v3执行生命周期公式和独占闲置缓存抵扣，释放缓存时同步提升未完成请求预算。

`RAG_MEMORY_PROBE_PROCESS=1`隔离同一device-context采样接口，避免模型allocator锁阻塞读数；实验采样间隔0.001秒，实际采样时间/时延另行保存，不能声称每1ms必定取得样本。`RAG_NPU_CACHE_BRIDGE`指定按当前torch/torch_npu/ABI和源码哈希构建的公开C++缓存回收桥，运行前校验元数据及二进制哈希。构建：`scripts/runtime/build_npu_cache_bridge.py`。

实验准备工具`scripts/runtime/restart_memory_api.py`默认只预览；`--run`要求无后台测试且已接收请求/预算均排空后重启API，保留现有模型与数据库参数。只能用于独立实验开始前准备，不能把重启后的显存下降作为滞回恢复证据。具体复现见[水位实验](watermarks-experiment.md)。
