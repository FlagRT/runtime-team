# OneRec 生成阶段

更新：2026-11-11。当前实测状态及证据以根目录 [status.md](../status.md) 为准。

## 模型与设备

模型为 `OpenOneRec/OneRec-8B`，下载元数据版本
`29f95b3da63a4bbc00a1cd3aeee948efb2ff40ff`，Qwen3ForCausalLM、BF16，
36层、8个KV heads、head_dim 128。三个权重分片合计16,782,026,752字节（约15.63GiB）。
宿主机路径 `/home/jliu171/jliu171_data/models/OneRec-8B`，容器内 `/mnt/onerec/OneRec-8B`。

用户最后指定改试**物理NPU卡0、chip0、die0、`/dev/davinci0`**，单die容器内使用`npu:0`。
此前物理卡7的die15已尝试，设备节点返回EBUSY；`/dev/davinci7`属于物理卡3。
新增overlay保留原embedding/reranker模型挂载及实验输出挂载，只增加OneRec只读挂载。
从宿主机项目目录使用：

```bash
RAG_NPU_DIE=0 docker compose --env-file .env -f ../compose.base.yml \
  -f docker_compose/compose.vllm.yml -f docker_compose/compose.onerec.yml \
  up -d --no-deps runtime-dev
```

该命令会在配置改变时重建主容器。仅在已有推理任务结束后执行。
overlay默认使用已保存运行环境镜像`rag-ljy-runtime:npu7-backup`，可用`RAG_RUNTIME_IMAGE`指定另一份已准备的运行镜像。
镜像tag中的`npu7`只是历史命名，实际设备由`RAG_NPU_DIE`选择；设15表示物理卡7/chip1。
原始厂商镜像不包含本地额外安装的完整环境，不能直接替换后假定依赖仍然齐全。

## 实现与接口

生成默认关闭。`--generation`启用本地Transformers + torch-npu模型；
仍采用原vLLM embedding、CPU数据库混合检索与Qwen3精排。
流程为向量化→检索→精排→受token预算限制的上下文→OneRec生成。
生成有独立有界队列，并与精排共享同一个初始化/执行线程，两种本地模型的forward不重叠。
多个请求仍可按各自依赖推进，默认等待队列容量8，不增加8请求的全局数量限制。

`POST /query`默认返回原有检索结果列表；设置`"generate":true`后返回
`answer`、实际用于prompt的`sources`、原精排`hits`、token用量、`finish_reason`、
`context_truncated`及阶段耗时。来源的`citation_id`与prompt中的`[1]`等编号对应。
模型是否正确引用/根据证据回答，仍需输出质量评测，不能仅凭接口成功认定。

默认请求最大输出256 token；服务上限1024 token，输入上限4096 token，均可显式配置。
上下文按精排顺序放入；超限时只裁剪后续证据，返回裁剪标志及实际使用文本。
问题本身过长则在接收前返回422，不悄悄裁剪问题。
使用模型原chat template，`enable_thinking=False`、greedy、batch 1、dynamic KV cache、
`logits_to_keep=1`，每次prefill仍处理完整prompt，每个输出token都真实解码。

## 显存规则

开启生成必须同时开启动态显存准入，禁止用固定预算或关闭准入运行生成服务。
模型常驻权重计入device-context的实测占用，不再次计入每请求新增预算。
新请求预算为embedding、精排、生成各阶段的保守新增峰值之和；
生成按配置允许的最大输入及该请求最大输出计算，精排减小batch时仍完整保留生成预算。

生成预算包含prefill/decode注意力、激活、完整最大输出长度的KV、动态KV拼接额外空间、
最终位置logits、workspace与分配器增长空间。保持dense注意力估算，未用未经验证的融合算子折扣。
这是`analytical_uncalibrated`解析估算，尚不是已校准的通用上界。
生成前按实际prompt长度复核实时显存及5GiB余量，整个生成执行观测本进程allocator峰值；
超过估算会给原全链路估算器置fault，停止后续准入。

原策略保持：`device-context.runtime.memory_stats()`是设备显存读数来源；
实测≥85%暂停、≤75%恢复；新请求须满足
`实测空闲−未释放预算−本次新增预算≥5120MiB`。
预算保留到生成/结果组织完成，并等待完成后开始的新有效采样才释放。
HTTP等待取消不取消已接收工作；拒绝返回503及Retry-After，不放回队列。

## 启动与检查

首先完成设备可见性与模型文件只读预检：

```bash
docker exec -w /workspace/dev/rag_ljy -e DC_BACKEND=ascend \
  -e PYTHONPATH=/workspace/dev/device-context/prototype \
  rag-ljy-vllm-910c .venv/bin/python scripts/checks/check_device.py --device npu:0

docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python scripts/checks/check_onerec_generation.py
```

独立单请求生成（不是数据库或85%验收）：

已有查询API运行时，该入口会拒绝加载第二份OneRec；应改用后面的`check_generation_live.py`检查现有服务。

```bash
docker exec -w /workspace/dev/rag_ljy -e DC_BACKEND=ascend \
  -e PYTHONPATH=/workspace/dev/device-context/prototype \
  -e RAG_MEMORY_PROBE_PROCESS=1 \
  -e RAG_NPU_CACHE_BRIDGE=/workspace/dev/rag_ljy/experiment_outputs/runtime_extensions/82f0cce02d30e7b44704/libnpu_cache_bridge.so \
  rag-ljy-vllm-910c .venv/bin/python -u scripts/checks/check_onerec_generation.py \
  --run --allow-unverified-scope
```

完整服务需要先按[运行指南](operations.md)启动选定数据库和8089 embedding引擎。
在容器内启动8088 API；以下是Milvus配置，换ES时改`RETRIEVAL_BACKEND=elasticsearch`：

```bash
cd /workspace/dev/rag_ljy
export DC_BACKEND=ascend
export PYTHONPATH=/workspace/dev/device-context/prototype
export RETRIEVAL_BACKEND=milvus
export MILVUS_COLLECTION=nfcorpus_chunks_v1
export ELASTICSEARCH_INDEX=nfcorpus-chunks-v1
export RAG_MEMORY_PROBE_PROCESS=1
export RAG_RERANK_DOCUMENT_TOKENS=640
export RAG_NPU_CACHE_BRIDGE=/workspace/dev/rag_ljy/experiment_outputs/runtime_extensions/82f0cce02d30e7b44704/libnpu_cache_bridge.so
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.70

.venv/bin/python -u scripts/runtime/serve_queries.py --device npu:0 --port 8088 \
  --generation --generation-model /mnt/onerec/OneRec-8B \
  --generation-max-input-tokens 4096 --generation-max-new-tokens 1024 \
  --max-inflight 0 --stage-queue-capacity 8 --embedding-workers 8 --retrieval-workers 4 \
  --max-reranker-batch-size 32 --memory-admission --memory-budget-mode dynamic \
  --memory-probe-interval 0.1 --memory-safety-mib 5120 \
  --memory-high-watermark .85 --memory-low-watermark .75 --memory-allow-unverified-scope
```

`--memory-allow-unverified-scope`显式接受runtime当前标记为`backend_reported_unverified`的统计范围；
不能据此认定所有卡商统计都为整卡视角。
等待`/health`中的生成模型已启用、显存准入为ready，再发送请求：

```bash
curl -f -H 'Content-Type: application/json' \
  -d '{"query":"How does nutrition affect health?","generate":true,"max_new_tokens":256}' \
  http://127.0.0.1:8088/query
```

三种到达模式的功能检查（容器内，默认仅预检，`--run`才发请求）：

```bash
.venv/bin/python scripts/checks/check_generation_live.py --backend milvus --run
```

## 原50条查询的完整工作流测试

输入仍是`data/queries/nfcorpus-50.jsonl`，SHA256为
`9b0a898ba4e50ccf0081591ba4a2a1ddccc960c112582692a52e90aca09e7429`。
脚本核对50个唯一query ID和完整文件校验值，按原文发送，每条都设置`generate:true`。
默认串行，每条最大生成256 token，等待上一条预算释放后再发送下一条；503不会自动重试。
只有50条全部获得完整生成结果、来源、token计数和阶段耗时才报告`passed`。
任何拒绝会列出对应query ID并标记`incomplete`，运行错误/缺失生成结果会标记`failed`。
答案内容、引用编号、原词复述及输出截断会记录，但这不是答案质量的自动认证。

宿主机执行，针对当前已运行的Elasticsearch API：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_generation_50.sh --backend elasticsearch --run
```

默认不带`--run`时只预检，不发送查询。可加`--parallel 2`运行有界并发批次，输入仍为50条各一次；
如出现预算拒绝，不会把少量成功请求当成50条全部完成。
`--backend`用于核对已有API，脚本不切换数据库、不重启服务、不加载额外模型。
Milvus需先按前面的启动配置运行对应API，再使用`--backend milvus`。

输出为`experiment_outputs/benchmarks/generation_50/<backend>/<UTC时间>/`，保留
`queries.jsonl`、逐请求`requests.jsonl`、实际显存时间线`memory.jsonl`、
`manifest.json`、源码快照及`summary.json`。汇总包括覆盖数、失败/未发送ID、token用量及耗时。
2026-11-11准备阶段通过9项模拟HTTP服务自测；随后用户执行Elasticsearch串行50条，
50/50 HTTP200、0次503、峰值48.252%，原始结果见根目录[status.md](../status.md)。
20条达到256输出token上限、1条复述输入，功能通过不代表答案质量或85%专项通过；Milvus全50条尚未执行。

## 包含生成的水位专项

包含生成的真实85%三模式专项（宿主机，要求现有API已开启生成和动态准入）：

```bash
bash scripts/checks/run_measured_acceptance.sh --backend milvus --generate --run \
  --warmup-profile 1024:32 --profiles 2048:32 --max-new-tokens 64
```

ES使用`--backend elasticsearch`。上述档位只是可复现输入配置，新增常驻模型后是否达到85%必须实测，
未达到会返回`incomplete`，不能沿用此前未含生成的成功结论，也不能为跨阈值而减小安全余量或估算系数。
该套件保留每个模式前的独立冷启动，在模式内通过正常回收验证恢复。

结果沿用`experiment_outputs`软链接指向的原实验盘：
`checks/onerec_generation/`、`checks/generation_integration/<backend>/`、
`checks/measured_watermarks_generation/<backend>/`、`checks/memory_acceptance_generation/<backend>/`。
记录输入、版本元数据、源码快照、实际显存、预算/HTTP与返回；NPU计算利用率未由这些检查测量。
