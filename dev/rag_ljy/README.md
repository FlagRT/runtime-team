# Elasticsearch/Milvus + Ascend NPU RAG（rag_ljy）

> **状态：🔄 可切换检索后端的代码已实现，待两个后端分别完成集成验证**

## 目标

构建一条可运行的 RAG 链路。`RETRIEVAL_BACKEND` 在 Elasticsearch 和 Milvus
之间进行完整数据库切换；被选中的数据库同时执行 BM25 稀疏检索和 dense
向量检索。`torch_npu` 在 Ascend NPU 上执行 embedding、rerank 和后续生成。

一次请求只使用一个数据库，不采用 Elasticsearch 做 sparse、Milvus 做 dense
的跨数据库组合。

## 已选组件

| 阶段 | 实现 |
|---|---|
| Sparse retrieval | 当前所选数据库的 BM25 |
| Dense retrieval | `Qwen/Qwen3-Embedding-0.6B`，默认 1024 维 |
| Coarse ranking | BM25 与 dense Top 50 的客户端 RRF，默认输出 Top 30 |
| Fine ranking | `Qwen/Qwen3-Reranker-0.6B`，默认输出 Top 5 |
| Generation | 暂未选择；按 NPU 显存和 FlagOS/vLLM 兼容性单独选择 |

## 整体工作流

### 文档入库

```text
原始文档
  -> 清洗和切块
  -> torch_npu embedding
  -> 当前所选数据库保存文本、元数据和向量
```

### 在线问答

```text
用户问题
  -> torch_npu 生成 query embedding
  -> 当前所选数据库执行 BM25 + dense vector 两路召回
  -> RRF coarse ranking（粗排），取 Top 20~50
  -> torch_npu reranker fine ranking（精排），取 Top 3~5
  -> 构造 prompt
  -> NPU 大模型生成答案
  -> 返回答案和引用来源
```

## 数据库切换语义

`.env` 中只需要选择一个后端：

```env
RETRIEVAL_BACKEND=elasticsearch
```

或：

```env
RETRIEVAL_BACKEND=milvus
```

两种模式分别为：

```text
elasticsearch: BM25 -> Elasticsearch，dense -> Elasticsearch
milvus:        BM25 -> Milvus，        dense -> Milvus
```

切换后端不会自动复制数据。首次使用某个后端时，必须对该后端分别执行
`create_index.py` 和 `ingest_test_data.py`。`chunk_id` 在两个后端保持一致，
便于后续进行结果和性能对比。

## 代码结构

```text
src/rag_engine/stores/
├── base.py             # 两个后端共同遵守的 RetrievalStore 接口
├── elasticsearch.py    # Elasticsearch BM25 + dense 实现
├── milvus.py           # Milvus BM25 + dense 实现
└── factory.py          # 根据 RETRIEVAL_BACKEND 创建一个后端
```

公共引擎只调用以下接口，不直接判断数据库类型：

```text
require_connection()
has_index()
create_index()
bulk_index()
bm25_search()
dense_search()
```

`retrieval.py` 使用同一个 store 完成 sparse 和 dense 检索，再执行 RRF。
embedding、RRF、reranker 和后续 generation 不随数据库切换。

## 数据约定

每条记录代表一个 chunk，两个后端保存相同的业务字段：

| 字段 | 用途 |
|---|---|
| `chunk_id` | chunk 唯一标识和主键 |
| `document_id` | 原文档标识 |
| `document_version` | 文档版本 |
| `chunk_index` | chunk 在文档中的顺序 |
| `title` | 标题检索和结果展示 |
| `text` | BM25 检索和 prompt 上下文 |
| `source_uri` | 答案引用来源 |
| `metadata` | 业务元数据 |
| `embedding` | dense vector 检索 |

Elasticsearch 使用 `rag-chunks-v1` 索引，Milvus 使用 `rag_chunks_v1` collection。
Milvus collection 额外维护 `search_text` 和 BM25 生成的
`sparse_embedding`；两者属于后端内部字段。

`EMBEDDING_DIMS` 必须与实际 embedding 输出维度一致。修改维度或 mapping/schema
后，需要使用新名称或明确执行 `create_index.py --recreate`。

## 容器划分

| 容器 | 设备 | 职责 |
|---|---|---|
| `rag-ljy-elasticsearch` | CPU | Elasticsearch BM25 和 dense 检索 |
| `rag-ljy-milvus` | CPU | Milvus BM25 和 dense 检索 |
| `flagos-rag-ljy-dev-910c` | Ascend NPU | RAG 代码、embedding、rerank 和 generation |

Elasticsearch 和 Milvus 都是独立的 CPU-only 容器，不映射 `/dev/davinci*`，
也不占用 NPU/DrvMng 客户端名额。两个数据库容器可以同时运行以便测试，
但一条 RAG 请求只访问 `RETRIEVAL_BACKEND` 指定的一个数据库。

NPU 开发容器来自 `../compose.base.yml` 并使用 host network，因此可以通过
`127.0.0.1:9200` 和 `127.0.0.1:19530` 访问宿主机映射的数据库端口。

## 目录约定

```text
dev/rag_ljy/
├── README.md
├── pyproject.toml
├── docker-compose.yml                 # NPU 开发容器覆盖配置
├── docker_compose/
│   ├── compose.elasticsearch.yml      # 独立 CPU-only Elasticsearch
│   └── compose.milvus.yml             # 独立 CPU-only Milvus
├── .env                               # 本地配置和密码，不入 Git
├── data/sample_documents.jsonl
├── scripts/                           # 建库、入库、查询和模型下载入口
├── src/rag_engine/                    # RAG 引擎
└── tests/                             # 不需要 NPU 的单元测试
```

## 启动容器

进入项目目录：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
```

启动 Elasticsearch：

```bash
docker compose --env-file .env \
  -f docker_compose/compose.elasticsearch.yml \
  up -d
```

启动 Milvus：

```bash
docker compose --env-file .env \
  -f docker_compose/compose.milvus.yml \
  up -d
```

查看状态：

```bash
docker ps --filter name=rag-ljy-elasticsearch
docker ps --filter name=rag-ljy-milvus
```

启动 NPU 开发容器：

```bash
docker compose --env-file .env \
  -f ../compose.base.yml \
  -f docker-compose.yml \
  up -d runtime-dev
```

进入 NPU 开发容器：

```bash
docker exec -it flagos-rag-ljy-dev-910c bash
```

## Python 环境

以下命令在 NPU 开发容器内执行：

```bash
cd /workspace/dev/rag_ljy
./scripts/setup_npu_env.sh
source .venv/bin/activate
python scripts/check_npu.py
```

`setup_npu_env.sh` 使用 Python 3.11 的 `--system-site-packages` 创建 venv，复用
镜像中与 CANN 匹配的 `torch` 和 `torch_npu`，并安装 Elasticsearch 与 Milvus
Python 客户端。它不会重新安装 PyTorch。

## 运行检索链路

当前 shell 可以临时覆盖 `.env` 中的选择。例如验证 Milvus：

```bash
export RETRIEVAL_BACKEND=milvus
```

验证 Elasticsearch 时改为：

```bash
export RETRIEVAL_BACKEND=elasticsearch
```

### Checkpoint 1：BM25

不加载模型，也不占用 NPU：

```bash
python scripts/create_index.py
python scripts/ingest_test_data.py --skip-embedding
python scripts/test_bm25.py "How are sparse and dense retrieval combined?"
```

Milvus schema 要求每条记录包含 dense vector，因此 `--skip-embedding` 会暂时写入
一个非零占位向量；后续完整入库会通过相同 `chunk_id` upsert 为真实向量。

不要随意使用 `create_index.py --recreate`。该参数会删除当前所选后端中的
索引或 collection 及其全部文档。

### Checkpoint 2：完整检索与精排

模型目录：

```text
/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B
/mnt/raid/jliu171/models/Qwen/Qwen3-Reranker-0.6B
```

运行：

```bash
python scripts/download_models.py
python scripts/ingest_test_data.py --device npu:0
python scripts/query.py \
  "How are sparse and dense retrieval combined?" \
  --device npu:0 \
  --coarse-top-k 30 \
  --fine-top-k 5
```

`query.py` 输出 BM25/dense rank、RRF score、reranker score、`document_id`、
`chunk_id` 和 `source_uri`，供后续 generation prompt 使用。

## 当前进度

| # | 任务 | 状态 |
|---|---|---|
| 1 | Elasticsearch Compose 服务 | ✅ |
| 2 | Milvus Compose 服务 | ✅ |
| 3 | 可切换的完整数据库后端 | ✅ |
| 4 | 两个后端分别完成 BM25 smoke test | 🔄 |
| 5 | NPU embedding、dense retrieval 和 RRF | 🔄 |
| 6 | NPU reranker fine ranking | 🔄 |
| 7 | Prompt 与 NPU generation | ⬜ |
| 8 | 两个后端的质量和延迟评测 | ⬜ |

> 状态图例：⬜ 待开始 ｜ 🔄 进行中 ｜ ✅ 完成

## 工作原则

- 每次查询只使用一个完整数据库后端。
- 两个后端使用相同的 chunking、embedding 模型、维度和 `chunk_id`。
- embedding、reranker 和生成模型及版本必须固定并记录。
- mapping/schema 变更使用新资源名，避免无意删除已有数据。
- 同一 Python 进程不同时加载 `torch_npu` 和 `torch_fl`。
- 不修改宿主机驱动或公共 NPU 配置。
- 多卡测试前先用 `npu-smi` 确认设备空闲。
