# 脚本入口

所有命令从 `dev/rag_ljy` 项目根目录执行。Python脚本通过共享 `_bootstrap.py`
读取本地 `.env` 并加载 `src/`；模型与数据库配置不需要写入脚本。

| 目录 | 常用入口 | 用途 |
|---|---|---|
| runtime/ | serve_embeddings.py、serve_queries.py、query.py | embedding引擎、API、单次查询 |
| runtime/ | setup_device_env.sh | 按镜像版本约束安装虚拟环境 |
| data/ | ingest_corpus_both.py、convert_beir_corpus.py、download_models.py | 数据转换、双库入库、模型准备 |
| benchmarks/ | benchmark_retrieval_comparison.py、benchmark_search.py | 顺序/并发检索对比与完整链路耗时 |
| benchmarks/ | check_queued_pipeline_200.py | 50查询×双库×双模式的完整API队列正确性测试 |
| benchmarks/ | prepare_benchmark_queries.py、check_baseline_200.py | 准备query集、重放历史与跨引擎对比 |
| checks/ | check_device.py、check_vllm_embeddings.py、verify_rebuilt_stores.py | 设备、向量与数据库核对 |
| checks/ | check_memory.py | 通过device-context采样设备内存，仅生成一个小型报告文件 |
| check_multi_request/ | check_requests.py | 串行/同时/错峰HTTP校验 |
| check_multi_request/ | capture_queue_evidence.py | 在线阶段时间线、队列深度、超限503与恢复证据采集 |
| diagnostics/ | diagnose_stages.py、diagnose_vllm_inputs.py、diagnose_rope.py | 故障复现与定位 |

结果统一写入 `artifacts/`，固定测试查询在 `data/queries/nfcorpus-50.jsonl`。
`check_npu.py`、`setup_npu_env.sh`、`benchmark_search.py` 保留各自已有别名语义。

详细参数运行入口的 `--help`；诊断脚本需按 [评测指南](../docs/evaluation.md)
启动匹配的服务后执行。路径迁移记录见 [docs/paths.md](../docs/paths.md)。
