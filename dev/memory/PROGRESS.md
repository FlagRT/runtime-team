# memory 子方向 · 项目进展时间线

> 更新：2026-09-03 ｜ 用途：子方向唯一追踪文档（待办 + 完成 + 时间线）；每项 ≤2 句、正文 ≤30 字
> 入口与操作：`README.md` ｜ 权威方案：《[显存与缓存管理方案-20260822](docs/路线A-显存与缓存管理-方案-20260822.md)》
> 历史/已冻结路线条目见 [docs/archive/README.md](docs/archive/README.md)，不作为当前进度或基线。

## 待办事项（按优先级）

| 优先级 | 事项 | 状态/依赖 |
|---|---|---|
| P0 | 目标模型清单确认（是否含混合注意力架构） | MoE 跨路线阻塞决定项，见 [昆仑芯问题反馈清单-20260822](docs/昆仑芯问题反馈清单-20260822.md) |
| 高 | 向智源/FlagOS 提交 issue（causal_conv1d / topk_softmax / moe_align_block_size / 文档滞后，共 5 项） | 附 file:line；清单见 [昆仑芯问题反馈清单-20260822](docs/昆仑芯问题反馈清单-20260822.md)；#5 已附根因与修复建议 |
| 中 | V3 分层缓存原型（KV 按需释放 + Host 溢出） | **P800（vllm 0.13）KV 卸载到 Host 已跑通**（09-01，官方 OffloadingConnector）；下一步：容量/驱逐行为/吞吐代价实测。**910C（vllm 0.20.2）官方 native 路径不可用**（09-03，is_cuda_alike 平台门 + vllm._C 缺 libcudart，见 [routeA-S4-KV卸载Host-910C尝试-20260903](docs/routeA-S4-KV卸载Host-910C尝试-20260903.md)）→ 昇腾需 plugin 侧补 CPU-offload handlers，或先敲定昇腾锁 0.13 还是 0.20.2 |
| 中 | 显存池定义与 V2 A/B 回归设计（vLLM 层） | 主战场 vLLM 层，厂商 torch 分配器为底座（无独立显存池） |
| 中 | 昇腾 venv 组合验证与 V1 画像 | 910c 机器补充（torch_npu + vllm 0.20.2） |
| 低 | V4 SSD 层评估（NVMe 带宽实测） | 随时可做 |
| 低 | 执行计划感知分配 | 等编译组接口答复，问题仍开放 |

## 完成事项（按时间，新→旧）

| 时间 | 事项 | 引用 |
|---|---|---|
| 09-03 | **S4 KV 卸载到 Host 移植 910C —— 阻塞留档**：vllm 0.20.2 官方 native `OffloadingConnector` 在昇腾栈不可用，硬阻塞两处：① `CPUOffloadingSpec.get_handlers()` 平台门 `is_cuda_alike()`（PlatformFL=npu→False，P800 xpytorch→True 故放行）② `vllm._C` CUDA 构建缺 `libcudart.so.13`→`swap_blocks_batch` 不存在。另：extra_config 键 0.20.2 改为 `cpu_bytes_to_use` | [routeA-S4-KV卸载Host-910C尝试-20260903](docs/routeA-S4-KV卸载Host-910C尝试-20260903.md) + probes/routeA_s4_kv_host_offload_910c.py |
| 09-01 | **纯 MoE 生成质量退化根因定位（issue #5）**：实锤厂商 `patch_decode_attention`（decode 无条件替换为 prefix-cache prefill_attention）为退化源（非 expert GEMM，dense 同退化），禁用后正常、解码提速近 2x | [新线栈decode生成退化-根因定位-20260901](docs/新线栈decode生成退化-根因定位-20260901.md) |
| 09-01 | **V3 第一步：KV 卸载到 Host 跑通（P800）**（0.13 官方 OffloadingConnector）：store/load 双向实测、吞吐代价 ~2.4%；num_cpu_blocks=0 接线缺陷已绕过（显式 KVTransferConfig） | [vllm-0.13-allocator与offload调研-20260822](docs/vllm-0.13-allocator与offload调研-20260822.md) §4 + probes/routeA_s4_* |
| 08-22 | 新线镜像纯 MoE 复测：topk_softmax 阻塞绕开（dispatch 降级 reference.torch），expert GEMM 首次触达；eager 可生成但质量退化，默认模式 graph capture 35min 不可用（部分解锁） | [新线镜像纯MoE复测-20260822](docs/新线镜像纯MoE复测-20260822.md) |
| 08-22 | P800 V1 显存画像：7 阶段全绿，KV 556,352 tokens/76.40GiB（~89%），910c P0 不存在 | [路线A-P800显存画像报告-20260822](docs/路线A-P800显存画像报告-20260822.md) |
| 08-22 | vllm 0.13 allocator/offload 调研：原生 KV CPU 卸载（--kv-offloading-size）发现 | [vllm-0.13-allocator与offload调研-20260822](docs/vllm-0.13-allocator与offload调研-20260822.md) |
| 08-22 | 三 issue 清单成形（causal_conv1d / topk_softmax / 回退开关失效） | [昆仑芯问题反馈清单-20260822](docs/昆仑芯问题反馈清单-20260822.md) |
| 08-22 | 官方镜像 MoE 复测 + 纯 MoE 隔离：定位 causal_conv1d / topk_softmax 双阻塞 | [官方镜像复测-MoE-20260822](docs/archive/官方镜像复测-MoE-20260822.md) · [纯MoE-昆仑芯-20260822](docs/archive/纯MoE-昆仑芯-20260822.md) |
| 08-21 | 昆仑芯 P800 可用性实测：官方栈端到端跑通 Qwen3-4B（96.5 tok/s），分层验证 + 阻塞项清单 | [路线A-P800可用性实测-20260821](docs/路线A-P800可用性实测-20260821.md) |

> 更早条目（2.4 方案定稿、分配器/V1 画像、推理链路闭环、算子三方联调模板、新机器 torch_fl 栈复现等）属已冻结路线，见 [docs/archive/README.md](docs/archive/README.md)。
