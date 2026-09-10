# infer910c A/B 汇总 —— 轴: `alloc-conf`

> UNTESTED —— pending 910C 锁定镜像验证。HBM 峰值真值请对齐 infer910c_hbm_sampler.py CSV（按 tag）。

| run | tag | status | load_s | warmup_s | measured_s | total_s | throughput_req_s | model_weights_gib | available_kv_cache_memory | gpu_kv_cache_size_tokens | maximum_concurrency | peak_memory_gib | non_torch_memory_gib | profiled_total_gpu_memory | driver_max_memory_reserved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alloc-conf_alloc-default | alloc-default | untested-pending-910c-validation | 25.129 | 0.228 | 0.202 | 34.455 | 316.83 | 1.1333 | 53.79 | 503552 | 61.47 | None | 0.1 | None | 0 |
| alloc-conf_alloc-expandable | alloc-expandable | untested-pending-910c-validation | 26.535 | 0.227 | 0.199 | 35.81 | 321.61 | 1.1333 | 53.79 | 503552 | 61.47 | None | 0.1 | None | 0 |
