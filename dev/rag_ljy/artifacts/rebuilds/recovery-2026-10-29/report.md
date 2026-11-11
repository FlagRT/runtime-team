# RAG 重新上线记录 — 2026-10-29（北京时间）

## 实际完成

- 复用宿主机已下载的 Qwen3-Embedding-0.6B、Qwen3-Reranker-0.6B 与 NFCorpus；未再次下载。
- 模型宿主机路径：`/home/jliu171/jliu171_data/hetero_data/models`，容器路径：`/mnt/raid/jliu171/models`。
- `.env` 保存新的宿主机模型路径；两份 RAG Compose 配置显式设置容器 `MODEL_DIR`，避免将宿主机路径误用于容器内模型加载。
- 修复重建磁盘后两个空数据库目录的属主：Milvus 999:999，Elasticsearch 1000:0。数据库继续使用本地 RAID 上原有两个挂载目录，没有迁移到模型所在共享文件系统。
- 恢复 Milvus 原镜像的丢失标签，仍是同一 image ID。Milvus 使用内嵌 etcd，仍然只有一个 Milvus 容器。
- 启动 vLLM embedding（8089），使用逻辑 npu:0 / 宿主机 `/dev/davinci1`；保留 `TRITON_ENABLE_TASKQUEUE=true`。
- 执行 `scripts/data/ingest_corpus_both.py`：3,633 篇文档、6,543 个分块、1024 维向量，同批向量写入两库。ES `nfcorpus-chunks-v1` 和 Milvus `nfcorpus_chunks_v1` 各验证到 6,543 条；两个数据库的 BM25 和 dense 检索均通过冒烟检查。
- 启动 RAG API（8088），后端 Milvus，vLLM embedding + Transformers NPU reranker，开启设备内存探测和准入。

## 验证与边界

- 两轮共 12 次真实 API 请求全部 HTTP 成功，结果与各轮串行参考一致（按现有校验器的排序、内容及分数容差）。
- 第一轮：串行2、同时2、间隔2秒2；间隔2秒的两个请求均成功，但没有重叠，校验器因要求 HTTP 重叠将该场景标为 failed；并非请求失败。保留原始报告，不改写为通过。
- 第二轮：串行2、同时2、间隔20毫秒2，三场景均通过并观察到同时/错峰的 HTTP 重叠。
- `/health` 为 ready，12 accepted / 0 rejected，完成后 inflight=0。
- 内存准入：85%/75% 水位、每请求2048 MiB逻辑预算、1024 MiB安全余量、每秒采样。当前允许 `backend_reported_unverified` scope，预算未经本次压力校准。
- 本轮是重建和上线冒烟验证，不是持续负载、内存不足拒绝/恢复或完整200次回归实验；也不据客户端重叠推断服务端阶段重叠。
- 仅宿主机本地接口可用；当前不提供答案生成。

## 使用

```bash
curl -s http://127.0.0.1:8088/health | python -m json.tool
curl -s http://127.0.0.1:8088/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What are the health benefits of green tea?"}'
```

模型/API 在 `rag-ljy-vllm-910c` 内通过 nohup 启动，日志为本目录 `embedding.log` 和 `query.log`，PID 分别记录在容器 `/tmp/rag-embedding-vllm.pid`、`/tmp/rag-query-server.pid`。容器重启后需重新启动这两个进程；没有新增自动启动管理器。

实际服务命令（在容器 `/workspace/dev/rag_ljy` 内，模型服务先就绪）：

```bash
export MODEL_DIR=/mnt/raid/jliu171/models
export DC_BACKEND=ascend RAG_EMBEDDING_BACKEND=vllm
export RETRIEVAL_BACKEND=milvus MILVUS_COLLECTION=nfcorpus_chunks_v1
export ELASTICSEARCH_INDEX=nfcorpus-chunks-v1
# 第一终端
.venv/bin/python scripts/runtime/serve_embeddings.py --device-index 0 --port 8089
# 第二终端：同样设置上述环境变量
.venv/bin/python scripts/runtime/serve_queries.py --device npu:0 --port 8088 \
  --memory-admission --memory-request-mib 2048 --memory-allow-unverified-scope
```

证据：`manifest.json`（输入哈希与配置）、`containers.txt`（镜像ID）、`ingestion.log`、`health.json`、`api-checks/`（两轮原始请求、返回结果及判定）。
