# 评测与固定输入

所有示例从 `/workspace/dev/rag_ljy` 执行，先激活 `.venv` 并启动所需服务。

## 同一组50条查询

固定输入：[data/queries/nfcorpus-50.jsonl](../data/queries/nfcorpus-50.jsonl)。
SHA256：`9b0a898ba4e50ccf0081591ba4a2a1ddccc960c112582692a52e90aca09e7429`。
这是从 NFCorpus test queries 选出的同一组50条；200次指50条×两个数据库×
两种模式，不是200条不同查询。脚本保持query ID和文本，不重新抽取其他输入。
原始语料3,633文档，当前两个后端各6,543块；模型为Qwen3-Embedding-0.6B。

## 顺序/并发检索200例

```bash
python scripts/benchmarks/benchmark_retrieval_comparison.py   --queries data/queries/nfcorpus-50.jsonl --device npu:0   --limit 50 --warmup 3 --repeats 1
```

默认结果写入 `artifacts/benchmarks/comparison-<UTC时间>-<ID>/`，包含
`report.md`、原始耗时、query vectors、queries和排序ID。两库分别运行；请求
按序进入，concurrent指同一请求的BM25/dense并行。向量在测量前计算并复用，
不包含embedding或精排耗时。`--modes concurrent sequential` 可反转顺序。
默认embedding为vLLM；需要8089引擎。`--backends milvus` 可仅测一个库。

[当前重建后的报告](../artifacts/rebuilds/vllm-rebuild-2026-10-09/comparison-20261009T045201Z-2a44ec/report.md)：
200/200成功，两库各50/50顺序/并发排名一致。当前检索均值为Milvus
7.126→4.246ms，Elasticsearch 18.653→11.535ms。共享主机单轮结果受缓存和负载影响。

[旧PyTorch报告](../artifacts/benchmarks/concurrent_vs_sequential_2026-09-23/comparison-20260923T111642Z-231050/report.md)
用于历史比较，向量对应重建前数据库；新文档向量生成后排名可能改变。

## 显式队列：完整API 200例

```bash
python scripts/benchmarks/check_queued_pipeline_200.py
```

该脚本使用同一固定50条查询，为每个数据库分别启动临时8090 API：sequential
检索模式下逐个发送50请求；concurrent检索模式下按2请求一组同时发送50请求。
共200次完整embedding→检索→精排API调用。默认worker为embedding=2、retrieval=4、
rerank=1；队列等待槽8，全局容量8。可用 `--parallel 4` 验证更高提交并发。

保存结果到 `artifacts/benchmarks/queued-pipeline-<日期时间>-<ID>/`，包含全部响应、
真实health配置、每阶段排队/执行耗时与服务端日志；对照串行参考核对请求ID、
结果归属、排序、内容与分数。采集服务器embedding阶段重叠证据。不是旧的
retrieval-only 200例耗时，且客户端并发与库内检索并发同时变化，不能据此单独
估计队列新增开销。脚本只关闭自己启动的临时API。

## 多请求API正确性

```bash
python scripts/check_multi_request/check_requests.py --backend milvus
python scripts/check_multi_request/check_requests.py --backend milvus --count 4 --stagger-ms 20
```

默认取固定输入的前两条，检查串行、同时到达和错峰到达；保存完整搜索结果到
`artifacts/checks/multi_request/<日期时间>-<backend>-<ID>/`。检查HTTP成功、
请求ID唯一、非空结果、顺序/内容与串行参考一致，分数有严格容差。
`--backend` 只是结果标签，实际后端由API进程设置；端口未启动会连接失败。
HTTP重叠并不证明服务内阶段重叠，需结合 `X-Request-ID` 与stage日志。
[当前vLLM API验证](../artifacts/checks/multi_request/2026-10-09_134804-milvus-174bdb/summary.json)
三种场景共6例通过。

## 在线队列与超限恢复证据

```bash
python scripts/check_multi_request/capture_queue_evidence.py
```

先启动带三阶段日志的8088 API，且确保当前没有其他测试请求。该脚本从health
读取实际数据库及全局准入上限，使用固定50查询的前8条作串行参考，再发送
8条同时到达、8条间隔20ms到达、9条超限突发（重复第一条参考查询）和1条恢复请求。
超限场景要求8条成功、1条503且Retry-After=1；每个接受请求的结果均与串行参考核对。
默认读取 `/tmp/rag-query-server.log`，通过 `--server-log` 可指定其他日志位置。

输出在 `artifacts/checks/queue_evidence/<日期时间>-<实际backend>-<ID>/`，保存可读
`report.md`、完整响应、health采样、服务端日志、JSON/CSV阶段时间线、源码快照与校验和。
[2026-10-09 实测报告](../artifacts/checks/queue_evidence/2026-10-09_140744-milvus-30d953/report.md)：
34次调用得到33次成功及1次预期拒绝；所有成功请求三阶段依赖正确，结果与参考一致；
观察到60对跨请求的阶段区间重叠，队列采样未超限，完成后队列归零。
另外保存22项现有调度/API测试的日志及JUnit记录，覆盖队列背压、取消、错误隔离等行为。
这验证的是数量准入控制；内存水位准入尚未实现，也没有证明实际vLLM forward batching。

## 其他检查

```bash
python scripts/checks/check_device.py --device npu:0
python scripts/checks/verify_rebuilt_stores.py --output /tmp/rag-store-verification.json
python scripts/checks/check_vllm_embeddings.py --count 2
python -m pytest tests -q
```

跨引擎向量检查比较9月23日PyTorch向量与vLLM输出，阈值cosine>=0.999，
结果在 `artifacts/checks/embeddings/`。这是兼容性检查，不是qrels质量评测。
重建后也不应期待与旧PyTorch排名完全一致。

`benchmark_search.py --scope full-query` 可记录embedding、检索、RRF、精排。
七项耗时为：embedding_ms、bm25_ms、dense_ms、search_ms、rrf_ms、rerank_ms、
total_ms；retrieval-only不测模型阶段。运行 `--help` 查看参数。

## 故障诊断与下游引用

诊断脚本在 `scripts/diagnostics/`；证据在 `artifacts/diagnostics/`。
`diagnose_vllm_inputs.py` 重放固定token IDs，`diagnose_stages.py` 捕获各阶段结果，
`diagnose_rope.py` 对RoPE核执行独立控制测试。需要匹配的NPU镜像与运行服务；
它们是诊断入口，不属于日常200例基准。

下游任务优先读取 [artifacts/index.json](../artifacts/index.json) 的稳定键及路径。
[artifacts/relocations.json](../artifacts/relocations.json) 保存旧→新目录映射。
历史raw JSON、日志与源码快照保留原始字节；其中的绝对路径记录当时运行位置，
不是当前入口。新入口、未来输出默认路径、文档命令均已切换到新布局。
