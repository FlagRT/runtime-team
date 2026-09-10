# infer910c A/B 汇总 —— 轴: `gpu-mem-util`

> UNTESTED —— pending 910C 锁定镜像验证。HBM 峰值真值请对齐 infer910c_hbm_sampler.py CSV（按 tag）。

| run | tag | status | load_s | warmup_s | measured_s | total_s | throughput_req_s | model_weights_gib | available_kv_cache_memory | gpu_kv_cache_size_tokens | maximum_concurrency | peak_memory_gib | non_torch_memory_gib | profiled_total_gpu_memory | driver_max_memory_reserved |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu-mem-util_gmu0.4 | gmu0.4 | untested-pending-910c-validation | 23.249 | 0.262 | 0.204 | 32.46 | 313.73 | 1.1333 | 23.15 | 216704 | 26.45 | None | 0.1 | None | 0 |
| gpu-mem-util_gmu0.9 | gmu0.9 | untested-pending-910c-validation | 24.914 | 0.212 | 0.198 | 34.17 | 323.23 | 1.1333 | 53.79 | 503552 | 61.47 | None | 0.09 | None | 0 |
