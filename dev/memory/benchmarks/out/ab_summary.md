# infer910c 显存池 A/B 对照汇总（Phase C / P0-4）

> 锁定镜像 vllm-ascend:v0.20.2rc1-a3，Qwen3-Embedding-0.6B，davinci-7，batch 64 / seq≈512 / warmup 3。
> HBM 峰值 = 宿主 npu-smi 采样（tag=phaseC）按 run 分段的 max；其余字段来自各 run 的 profile JSON / vLLM 日志。

| axis | point | hbm_peak_mib | hbm_peak_pct | kv_pool_gib | kv_tokens | max_concurrency | weights_gib | peak_activation_gib_vllm | npu_graph_gib | non_torch_gib | load_s | warmup_s | measured_s | throughput_req_s | enforce_eager | alloc_conf | gpu_mem_util |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gpu-mem-util | gmu0.4 | 28072 | 42.8 | 23.15 | 216704 | 26.45 | 1.1333 | 0.22 | 0.1 | 0.1 | 23.249 | 0.262 | 0.204 | 313.73 | False | (default) | 0.4 |
| gpu-mem-util | gmu0.9 | 59689 | 91.1 | 53.79 | 503552 | 61.47 | 1.1333 | 0.22 | 0.09 | 0.09 | 24.914 | 0.212 | 0.198 | 323.23 | False | (default) | 0.9 |
| enforce-eager | aclgraph | 59671 | 91.1 | 53.79 | 503552 | 61.47 | 1.1333 | 0.22 | 0.1 | 0.1 | 27.526 | 0.219 | 0.202 | 316.83 | False | (default) | 0.9 |
| enforce-eager | eager | 59444 | 90.7 | 53.76 | 503296 | 61.44 | 1.1333 | 0.24 | 0.0 | 0.0 | 12.146 | 0.394 | 0.218 | 293.58 | True | (default) | 0.9 |
| alloc-conf | alloc-default | 59551 | 90.9 | 53.79 | 503552 | 61.47 | 1.1333 | 0.22 | 0.1 | 0.1 | 25.129 | 0.228 | 0.202 | 316.83 | False | (default) | 0.9 |
| alloc-conf | alloc-expandable | 59451 | 90.7 | 53.79 | 503552 | 61.47 | 1.1333 | 0.22 | 0.1 | 0.1 | 26.535 | 0.227 | 0.199 | 321.61 | False | expandable_segments:True | 0.9 |

## 要点

- **gpu-mem-util 0.4 vs 0.9 是唯一大杠杆**：KV/pooling 池 23.15 → 53.79 GiB，HBM 峰值 28072 → 59689 MiB（42.8% → 91.1%），Δ≈30.9 GiB ≈ Δgmu×空闲HBM。峰值 ≈ gmu×空闲HBM + ~4.5 GiB 固定（权重+激活+graph+driver）。
- **enforce-eager（eager）vs ACLGraph：显存差仅 ~0.2 GiB**（59444 vs 59671 MiB，= vLLM 报的 0.1 GiB NPU graph + rounding）。代价在别处：ACLGraph load +15s（torch.compile 26s），换 ~8% 吞吐（316 vs 293 req/s）。ACLGraph 在此 embedding 模型上**不是显存问题**（不同于旧栈生成式 graph 模式吃 ~7GiB）。
- **alloc-conf expandable_segments:True vs 默认：无可测差异**（59451 vs 59551 MiB，采样噪声内；KV 池、non_torch 均相同）。原因：KV/pooling 池是一次性整块分配，无增量 segment 增长可压缩；激活极小。**embedding 无碎片问题**。
