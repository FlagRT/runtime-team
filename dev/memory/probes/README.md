# probes/ —— 探针与画像脚本索引

> 只读诊断脚本，不改造子库；待入库的正式资产在此暂存。
> 运行环境见各脚本头 docstring。

## 当前方向（FlagOS 官方栈）

| 脚本 | 用途 | 平台 |
|---|---|---|
| `routeA_s2_1_device.py` | 设备枚举/初始化冒烟 | P800 |
| `routeA_s2_3_allreduce.py` | 双卡 allreduce 数值 | P800 |
| `routeA_s3_offline.py` / `routeA_s3_serve_client.sh` | dense 推理离线/服务化 | P800 |
| `routeA_s3_moe_ab.py` | MoE A/B（flag_gems vs reference） | P800 |
| `ref_moe_impl.py` | MoE 参考实现 | 通用 |
| `routeA_s4_kv_host_offload.py` / `routeA_s4_kv_offload_xfer.py` | KV→Host 卸载 + 传输冒烟 | P800（vllm 0.13） |
| `routeA_s4_kv_host_offload_910c.py` | KV→Host 卸载移植尝试（阻塞留档） | 910C（vllm 0.20.2） |
| `p800_env_check.py` / `p800_chain_smoke.py` / `p800_v1_memory_profile.py` | P800 环境/链路/V1 画像 | P800 |
| `fetch_qwen3_30b.py` | 模型下载工具 | 通用 |

## 通信相关（跨方向复用）

| 脚本 | 用途 | 备注 |
|---|---|---|
| `flagcx_smoke.py` | FlagCX 双卡 allreduce 冒烟（含异步返回需设备同步的现状） | communication 子方向《最小 Backend 契约》引用 |
| `hccl_direct.py` | 纯 ctypes HCCL 对照 | 与 `flagcx_smoke.py` 配套 |

## ⛔ 已冻结路线（torch_fl 设备层栈，仅留档，勿在其上继续开发）

> 这些脚本产生于昇腾 910C 的 torch_fl + vllm-plugin-FL 栈复现，该路线已冻结（见 [../docs/archive/README.md](../docs/archive/README.md)）。
> 保留仅供方法论/历史对照；**不要以它们为基线扩展新工作**，当前方向从 [../docs/路线A-显存与缓存管理-方案-20260822.md](../docs/路线A-显存与缓存管理-方案-20260822.md) 起步。

| 脚本 | 原用途 |
|---|---|
| `flagos_boot.py` | torch_fl venv 引导 shim（npu/cann→flagos 别名等；补丁台账见 archive/patches/） |
| `c10_npu_shim.cpp` | torch_npu 符号 stub（已被 FlagCX fix 分支取代） |
| `probe_allocator_profile.py` | torch_fl caching allocator 画像 |
| `qwen3_mini_probe.py` / `qwen3_offline_tp.py` | 910C torch_fl 栈推理闭环诊断 |
| `op_smoke.py` / `triton_smoke.py` / `triton_mm_smoke.py` / `triton_mm_debug.py` | 910C torch_fl 栈算子级隔离 |
| `linear_shape_probe.py` / `linear_twice.py` / `matmul_compare.py` | 同上（linear/matmul 数值与耗时对照） |
