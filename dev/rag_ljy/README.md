# Elasticsearch/Milvus + Ascend NPU RAG（rag_ljy）

更新日期：2026-09-11

## 已完成进展

- **Elasticsearch 检索精排链路已跑通**：完成 1024 维索引创建、样本文档向量化、BM25 + dense 检索、RRF 融合及 Qwen3 精排。5 条样本文档返回 Top 5，问题 `How does hybrid search work?` 对应文档排名第 1，rerank score 为 0.99944717。
- **常驻查询 API 已实现**：两个模型启动时加载一次、请求间复用，无空闲卸载；支持健康检查、参数校验及并发保护。NPU 连续请求与显存常驻待实跑验证。
- **双后端接口和单镜像环境已对齐**：Elasticsearch / Milvus 共用检索与精排流程；Milvus 集成实跑待完成。RAG 模型容器统一使用 vLLM Ascend 镜像，安装脚本按镜像推理栈版本约束依赖。

详细进展及实际排序结果见 [status.md](status.md)。当前返回排序后的文档片段；生成模型尚未接入。

## 目标

构建一条可运行的 RAG 链路。`RETRIEVAL_BACKEND` 在 Elasticsearch 和 Milvus
之间进行完整数据库切换；被选中的数据库同时执行 BM25 稀疏检索和 dense
向量检索。`torch_npu` 在 Ascend NPU 上执行 embedding 和 rerank；答案生成属于后续工作。

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

### 当前查询链路

```text
用户问题
  -> torch_npu 生成 query embedding
  -> 当前所选数据库执行 BM25 + dense vector 两路召回
  -> RRF coarse ranking（粗排），取 Top 20~50
  -> torch_npu reranker fine ranking（精排），取 Top 3~5
  -> 返回排序片段 JSON（文本、来源、检索排名、RRF / rerank 分数）
```

后续将基于精排片段构造 prompt，接入 generation 模型生成答案和引用。

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

## 容器与运行镜像

Embedding 和 reranking 使用同一个 NPU 容器，唯一模型运行镜像为：

```text
quay.io/ascend/vllm-ascend:v0.20.2rc1-a3
```

Python 3.11.15、PyTorch、torch_npu 和 vLLM 由镜像提供。当前模型通过
Transformers 直接推理，常驻 API 由 FastAPI 提供，无需启动 vLLM 模型服务器。

| 容器 | 设备 | 职责 |
|---|---|---|
| `rag-ljy-elasticsearch` | CPU | Elasticsearch BM25 和 dense 检索 |
| `rag-ljy-milvus` | CPU | Milvus BM25 和 dense 检索 |
| `rag-ljy-vllm-910c` | Ascend NPU | RAG 代码、embedding、rerank；后续接入 generation |

Elasticsearch 和 Milvus 都是独立的 CPU-only 容器，不映射 `/dev/davinci*`，
也不占用 NPU/DrvMng 客户端名额。两个数据库容器可以同时运行以便测试，
但一条 RAG 请求只访问 `RETRIEVAL_BACKEND` 指定的一个数据库。

NPU 开发容器来自 `../compose.base.yml` 并使用 host network，因此可以通过
`127.0.0.1:9200` 和 `127.0.0.1:19530` 访问宿主机映射的数据库端口。

## 目录约定

```text
dev/rag_ljy/
├── README.md
├── status.md                          # 已完成进展与实际结果
├── pyproject.toml
├── docker-compose.yml                 # NPU 开发容器覆盖配置
├── docker_compose/
│   ├── compose.elasticsearch.yml      # 独立 CPU-only Elasticsearch
│   ├── compose.milvus.yml             # 独立 CPU-only Milvus
│   └── compose.vllm.yml               # 唯一 NPU 模型容器
├── .env                               # 本地配置和密码，不入 Git
├── data/sample_documents.jsonl
├── scripts/
│   ├── setup_npu_env.sh               # 镜像依赖约束与 venv 安装
│   ├── create_index.py               # 创建选定后端的索引 / collection
│   ├── ingest_test_data.py           # 文档切块、embedding 和入库
│   ├── query.py                      # 单次查询，退出后释放模型
│   └── serve_queries.py              # 常驻 HTTP 查询，复用模型
├── src/rag_engine/                    # 检索引擎、server.py 和 query_runtime.py
└── tests/                             # 不需要 NPU 的单元测试
```

## 启动容器（宿主机执行）

进入项目目录：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
```

确认项目目录中的 `.env` 已配置 `ELASTIC_PASSWORD`，以及所需的数据库端口、
存储目录等参数。以下 `compose up` 命令会创建尚不存在的容器，也可启动已有服务。

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
docker ps -a --filter name=rag-ljy-elasticsearch
docker ps -a --filter name=rag-ljy-milvus
```

启动唯一的 NPU 容器（vLLM Ascend 镜像，自带 torch_npu）：

```bash
docker compose --env-file .env \
  -f ../compose.base.yml \
  -f docker_compose/compose.vllm.yml \
  up -d runtime-dev
```

进入 NPU 开发容器：

```bash
docker exec -it rag-ljy-vllm-910c bash
```

## Python 环境（容器内执行）

将 `N` 替换为已分配的 NPU 索引，后续入库和查询复用同一 `RAG_DEVICE`：

```bash
cd /workspace/dev/rag_ljy
bash scripts/setup_npu_env.sh
source .venv/bin/activate
export RAG_DEVICE=npu:N
python scripts/check_npu.py --device "$RAG_DEVICE"
```

`setup_npu_env.sh` 默认使用镜像 PATH 中的 `python3`（要求 >=3.11），先检查
`torch` / `torch_npu` 可以导入，再用 `--system-site-packages` 创建 venv，复用
镜像中与 CANN 匹配的 `torch` 和 `torch_npu`，并安装 Elasticsearch 与 Milvus
Python 客户端、RAG 模型依赖及 API 依赖。

`.venv` 位于挂载的仓库目录，因此文件保存在宿主机，但应使用容器内 Python 运行。
新开容器 shell 时重新执行 `source .venv/bin/activate`，并设置 `RAG_DEVICE`；
环境已就绪时无需每次重跑安装脚本。

安装时按镜像内的版本约束 PyTorch、torch_npu、Transformers、Hugging Face Hub、
tokenizers 和 vLLM，避免 RAG 依赖覆盖镜像的推理栈；最后执行 `pip check`。
若旧版脚本已在 `.venv` 中安装 Transformers 4.x / Hub 0.x，重新运行更新后的
`bash scripts/setup_npu_env.sh` 即可按镜像版本修复，已激活 `.venv` 时也可运行。

只启动一个 NPU 容器。`docker-compose.yml` 也指向同一镜像和容器名，是另一启动入口，
不要同时启动两个 Compose project。当前 embedding/rerank 直接调用 Transformers，
不需要启动 vLLM server；generation 模型、server 和检索结果到生成接口的连接尚未实现。
`python scripts/check_npu.py --device npu:N` 可检查已分配的设备；后续入库和查询也传
相同的 `--device npu:N`。默认 `npu:0` 不代表已经获得该设备的使用分配。

## 运行检索链路

当前 shell 可以临时覆盖 `.env` 中的选择。例如验证 Milvus：

```bash
export RETRIEVAL_BACKEND=milvus
```

验证 Elasticsearch 时改为：

```bash
export RETRIEVAL_BACKEND=elasticsearch
```

### 可选：仅验证 BM25

不加载模型，也不占用 NPU：

```bash
python scripts/create_index.py
python scripts/ingest_test_data.py --skip-embedding
python scripts/test_bm25.py "How are sparse and dense retrieval combined?"
```

Milvus schema 要求每条记录包含 dense vector，因此 `--skip-embedding` 会暂时写入
一个非零占位向量；后续完整入库会通过相同 `chunk_id` upsert 为真实向量。
该步骤仅用于首次排查，完整向量入库后不要再对同一数据执行 `--skip-embedding`，
否则会覆盖已写入的真实向量。

不要随意使用 `create_index.py --recreate`。该参数会删除当前所选后端中的
索引或 collection 及其全部文档。

### 完整入库与单次查询

模型目录：

```text
/mnt/raid/jliu171/models/Qwen/Qwen3-Embedding-0.6B
/mnt/raid/jliu171/models/Qwen/Qwen3-Reranker-0.6B
```

若尚未下载，运行 `python scripts/download_models.py --model all`；已下载时直接使用本地模型。
当前 `data/sample_documents.jsonl` 包含 5 条样本文档。

以下命令在容器内执行，先建库、入库，再查询：

```bash
python scripts/create_index.py
python scripts/ingest_test_data.py --device "$RAG_DEVICE"
python scripts/query.py \
  "How are sparse and dense retrieval combined?" \
  --device "$RAG_DEVICE" \
  --coarse-top-k 30 \
  --fine-top-k 5
```

`query.py` 输出 BM25/dense rank、RRF score、reranker score、`document_id`、
`chunk_id` 和 `source_uri`，供后续 generation prompt 使用。

## 常驻查询 API（模型只加载一次）

`query.py` 仍是单次查询命令，退出时释放模型。需要连续查询时，在现有
`rag-ljy-vllm-910c` 容器中运行下面的服务。它通过 Transformers/torch_npu
加载 embedding 和 reranker，并不是 vLLM generation server。

以下在现有容器内执行。先完成前面的环境安装和完整入库；将 `N` 替换为已分配的设备索引：

```bash
cd /workspace/dev/rag_ljy
source .venv/bin/activate
export RAG_DEVICE=npu:N
export RETRIEVAL_BACKEND=elasticsearch  # 或 milvus，须已入库
python scripts/serve_queries.py --device "$RAG_DEVICE" --port 8088
```

保持服务运行，从宿主机另一终端发送请求（容器使用 host network）：

```bash
curl --fail-with-body http://127.0.0.1:8088/health
curl --fail-with-body http://127.0.0.1:8088/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"How does hybrid search work?"}'
curl --fail-with-body http://127.0.0.1:8088/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What happens after reranking?","fine_top_k":3}'
```

返回值与 `query.py` 相同，是按精排分数排序的 JSON 数组。模型常驻至服务进程退出，
无空闲卸载；按 Ctrl+C 或向该进程发送 SIGTERM 可停止。异常退出、容器停止或设备
故障也会结束常驻，不能保证模型在进程退出后保留。

若要离开终端后继续服务，可在**容器内**使用以下命令替代前台启动（不要同时启动）：

```bash
nohup .venv/bin/python -u scripts/serve_queries.py --device "$RAG_DEVICE" --port 8088 \
  > /tmp/rag-query-server.log 2>&1 < /dev/null &
echo $! > /tmp/rag-query-server.pid
```

查看日志和显式停止也在容器内执行：

```bash
tail -n 100 /tmp/rag-query-server.log
kill -TERM "$(cat /tmp/rag-query-server.pid)"
```

服务仅绑定 `127.0.0.1`，没有应用层认证。对外提供服务应通过已有的鉴权反向代理。
固定使用一个 worker，在同一专用线程初始化和执行 NPU 模型；不要增加 worker 或
启用 reload，否则会重复加载模型。同一时刻处理一个查询，忙时返回 HTTP 503
及 `Retry-After: 1`，客户端应退避重试；没有无限增长的推理队列。
普通查询异常返回 HTTP 500 并记入日志，后续请求仍可提交；设备级故障可能需要
显式重启。请求可设置与 CLI 对应的 top-k 和 batch 参数，设备与数据库在启动时固定。
此服务完成的是持久化查询能力，生产并发容量仍需在目标 NPU 上压测。

## 后续工作

- 验证常驻 API 的连续查询、空闲显存保持及显式停止后的资源释放。
- 完成 Milvus 入库与查询实跑，比较两个后端的排序结果。
- 扩充标注数据集，测量检索质量、P50 / P95 延迟、QPS 和显存峰值。
- 接入 generation 模型，补齐基于检索片段的回答生成。

## 工作原则

- 每次查询只使用一个完整数据库后端。
- 两个后端使用相同的 chunking、embedding 模型、维度和 `chunk_id`。
- embedding、reranker 和生成模型及版本必须固定并记录。
- mapping/schema 变更使用新资源名，避免无意删除已有数据。
- 同一 Python 进程不同时加载 `torch_npu` 和 `torch_fl`。
- 不修改宿主机驱动或公共 NPU 配置。
- 多卡测试前先用 `npu-smi` 确认设备空闲。
