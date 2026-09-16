# probes/ —— 探针与画像脚本索引

> 只读诊断脚本，不改造子库；待入库的正式资产在此暂存。
> 运行环境见各脚本头 docstring。
> 按芯片分类：`common/`（跨芯片通用）→ `910c/` → `p800/` → `legacy-2.4-910c/`（⛔ 冻结路线，仅留档）。

## common/（跨芯片通用）

| 脚本 | 用途 | 备注 |
|---|---|---|
| `model-fetch_qwen3.py` | 模型下载工具（hf-mirror 直连 + 断点续传 + 并行 + 大小校验） | 通用 |
| `moe-ref-impl.py` | MoE 参考实现（A/B 对照用） | 通用 |
| `comm-smoke_flagcx.py` | FlagCX 双卡 allreduce 冒烟（含异步返回需设备同步的现状） | communication 子方向《最小 Backend 契约》引用 |

## 910c/（910C 原型阶段，2026-09，锁定推理镜像 `vllm-ascend:v0.20.2rc1-a3`，torch_npu 纯栈）

> 探针已实测通过（画像报告已交付，见 `../docs/goals/proto-910c-202609/`）。

| 脚本 | 用途 | 运行位置 |
|---|---|---|
| `hbm-sampler_910c.py` | 外挂 npu-smi 轮询采样 per-chip HBM/AICore → 时间戳 CSV（画像真值来源） | 宿主机（不进容器，无 torch 依赖） |
| `mem-profile_910c.py` | 锁定镜像内 torch_npu 口径显存画像 harness（offline/server-probe，pooling runner）→ JSON | 容器 `flagos-proto-infer-910c` 内 |
| `mem-ab-matrix_910c.py` | 跨 gpu-mem-util / alloc-conf / enforce-eager / max-num-seqs 轴批量跑 profile → 汇总 CSV+MD | 容器内 |
| `kv-offload-host_910c.py` | KV→Host 卸载移植尝试（阻塞留档） | 910C（vllm 0.20.2） |
| `comm-hccl-direct_910c.py` | 纯 ctypes HCCL 对照 | 与 `../common/comm-smoke_flagcx.py` 配套 |

## p800/（当前方向，FlagOS 官方栈）

| 脚本 | 用途 | 平台 |
|---|---|---|
| `device-smoke_p800.py` | 设备枚举/初始化冒烟 | P800 |
| `allreduce-smoke_p800.py` | 双卡 allreduce 数值 | P800 |
| `offline-infer_p800.py` / `serve-client_p800.sh` | dense 推理离线/服务化 | P800 |
| `moe-ab_p800.py` | MoE A/B（flag_gems vs reference，配合 `../common/moe-ref-impl.py`） | P800 |
| `kv-offload-host_p800.py` / `kv-offload-xfer_p800.py` | KV→Host 卸载 + 传输冒烟 | P800（vllm 0.13） |
| `env-check_p800.py` / `chain-smoke_p800.py` / `mem-profile-v1_p800.py` | P800 环境/链路/V1 画像 | P800 |
| `hbm-sampler_p800.sh` | 宿主侧 HBM/util 采样（对应 910c npu-smi 方法学，原 `benchmarks/xpu_smi_sampler.sh`） | P800 |

## ⛔ legacy-2.4-910c/（已冻结路线，torch_fl 设备层栈，仅留档，勿在其上继续开发）

> 这些脚本产生于昇腾 910C 的 torch_fl + vllm-plugin-FL 栈复现，该路线已冻结（见 [../docs/goals/legacy-2.4-910c/README.md](../docs/goals/legacy-2.4-910c/README.md)）。
> 保留仅供方法论/历史对照；**不要以它们为基线扩展新工作**，当前方向从 [../docs/common/design_显存与缓存管理权威方案.md](../docs/common/design_显存与缓存管理权威方案.md) 起步。

| 脚本 | 原用途 |
|---|---|
| `boot-shim_910c.py` | torch_fl venv 引导 shim（npu/cann→flagos 别名等；补丁台账见 `../docs/goals/legacy-2.4-910c/patches/`） |
| `c10-npu-shim_910c.cpp` | torch_npu 符号 stub（已被 FlagCX fix 分支取代） |
| `allocator-profile_910c.py` | torch_fl caching allocator 画像 |
| `qwen3-mini-probe_910c.py` / `qwen3-offline-tp_910c.py` | 910C torch_fl 栈推理闭环诊断 |
| `op-smoke_910c.py` / `triton-smoke_910c.py` / `triton-mm-smoke_910c.py` / `triton-mm-debug_910c.py` | 910C torch_fl 栈算子级隔离 |
| `linear-shape-probe_910c.py` / `linear-twice_910c.py` / `matmul-compare_910c.py` | 同上（linear/matmul 数值与耗时对照） |
