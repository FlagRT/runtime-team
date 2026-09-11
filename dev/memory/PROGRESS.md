# memory 子方向 · 项目进展时间线

> 更新：2026-09-10 ｜ 用途：子方向唯一追踪文档（待办 + 完成 + 时间线）；每项 ≤2 句、正文 ≤30 字
> 入口与操作：`README.md` ｜ 速览：`STATUS.md` ｜ 权威方案：《[显存与缓存管理方案-20260822](docs/common/design_显存与缓存管理权威方案.md)》
> 历史/已冻结路线条目见 [docs/goals/legacy-2.4-910c/README.md](docs/goals/legacy-2.4-910c/README.md)，不作为当前进度或基线。

## 910C 原型阶段 (2026-09) 范围调整

> ⚠️ **本期按 [运行时层原型验证-战略目标-910C.v1](../../docs/运行时层原型验证-战略目标-910C.v1.md) §3/§5/§7 重新收敛范围**。下方历史「待办/完成」表全部保留；与本期无关的行就地加「（本期冻结）」标注，不删除。

- **本期范围** = 910C 单芯片 + 锁定推理镜像 `vllm-ascend:v0.20.2rc1-a3`（华为昇腾官方纯栈，torch_npu）+ 验收模型 `Qwen/Qwen3-Embedding-0.6B`（embedding，非生成式）。
- **冻结（本期不作为进度/交付）**：全部 P800/昆仑芯条目；torch_fl / vllm-plugin-FL 的 910C dev 容器画像；MoE 阻塞链（causal_conv1d / topk_softmax / moe_align_block_size）；910C native KV→Host 卸载阻塞（routeA-S4 留档即可）；5 项 FlagOS/智源 issue 不卡本期验收。
- **本期 P0 待办**：
  1. 起 `flagos-proto-infer-910c` 跑通 embedding 推理（与 device-context 对齐，战略文档段二）。
  2. torch_npu 口径显存画像报告（加载阶段结构 + 运行阶段峰值），骨架见 [docs/goals/proto-910c-202609/profile_显存画像_910c.md](docs/goals/proto-910c-202609/profile_显存画像_910c.md)。
  3. 显存池定义文档（torch_npu caching allocator 底座 + vLLM 层：gpu_memory_utilization / KV-or-pooling 预分配 / ACLGraph capture）+ A/B 对照数据。
- **本期 P1**：维护 STATUS.md；给 device-context / 调度 输出安全 `gpu_mem_util` + `max_num_seqs` 区间；与监控共用一份 HBM + allocator 采样脚本（`probes/910c/hbm-sampler_910c.py`）；数据供 performance 统一验收报告。
- **已就绪（未测，pending 带卡容器 slot）**：`probes/910c/hbm-sampler_910c.py`（宿主 npu-smi 采样）、`probes/910c/mem-profile_910c.py`（容器内 torch_npu 画像 harness）、`probes/910c/mem-ab-matrix_910c.py`（A/B 矩阵驱动）。旧 flagos/xpytorch 探针（`p800/mem-profile-v1_p800.py` / `legacy-2.4-910c/allocator-profile_910c.py`）在锁定镜像 API 不存在，不复用。

## 待办事项（按优先级）

| 优先级 | 事项 | 状态/依赖 |
|---|---|---|
| P0 | 目标模型清单确认（是否含混合注意力架构） | **（本期冻结）** 本期验收模型锁定 `Qwen/Qwen3-Embedding-0.6B`；MoE 跨路线阻塞决定项，见 [昆仑芯问题反馈清单-20260822](docs/goals/bringup-p800/issue_昆仑芯问题反馈清单_p800.md) |
| 高 | 向智源/FlagOS 提交 issue（causal_conv1d / topk_softmax / moe_align_block_size / 文档滞后，共 5 项） | **（本期冻结，不卡本期验收）** 附 file:line；清单见 [昆仑芯问题反馈清单-20260822](docs/goals/bringup-p800/issue_昆仑芯问题反馈清单_p800.md)；#5 已附根因与修复建议 |
| 中 | V3 分层缓存原型（KV 按需释放 + Host 溢出） | **（本期冻结，routeA-S4 留档即可）** **P800（vllm 0.13）KV 卸载到 Host 已跑通**（09-01，官方 OffloadingConnector）；下一步：容量/驱逐行为/吞吐代价实测。**910C（vllm 0.20.2）官方 native 路径不可用**（09-03，is_cuda_alike 平台门 + vllm._C 缺 libcudart，见 [routeA-S4-KV卸载Host-910C尝试-20260903](docs/goals/proto-910c-202609/note_KV卸载Host尝试_910c.md)）→ 昇腾需 plugin 侧补 CPU-offload handlers，或先敲定昇腾锁 0.13 还是 0.20.2 |
| P0（本期） | 显存池定义文档 + A/B 对照数据（910C 锁定推理镜像口径） | 由「显存池定义与 V2 A/B」转本期口径：torch_npu caching allocator 底座 + vLLM 层（gpu_memory_utilization / KV-or-pooling 预分配 / ACLGraph capture）；脚本 `probes/910c/mem-ab-matrix_910c.py` 就绪，**阻塞：带卡容器 slot** |
| P0（本期） | 910C 锁定镜像显存画像报告（torch_npu 口径） | 由「昇腾 venv 组合验证与 V1 画像」转本期口径；环境侦察完成、探针移植未测、骨架 [docs/goals/proto-910c-202609/profile_显存画像_910c.md](docs/goals/proto-910c-202609/profile_显存画像_910c.md) 已建，**阻塞：带卡容器 slot（并发上限 3 已满）** |
| 低 | V4 SSD 层评估（NVMe 带宽实测） | **（本期冻结）** 随时可做 |
| 低 | 执行计划感知分配 | **（本期冻结）** 等编译组接口答复，问题仍开放 |

## 完成事项（按时间，新→旧）

| 时间 | 事项 | 引用 |
|---|---|---|
| 09-03 | **S4 KV 卸载到 Host 移植 910C —— 阻塞留档**：vllm 0.20.2 官方 native `OffloadingConnector` 在昇腾栈不可用，硬阻塞两处：① `CPUOffloadingSpec.get_handlers()` 平台门 `is_cuda_alike()`（PlatformFL=npu→False，P800 xpytorch→True 故放行）② `vllm._C` CUDA 构建缺 `libcudart.so.13`→`swap_blocks_batch` 不存在。另：extra_config 键 0.20.2 改为 `cpu_bytes_to_use` | [routeA-S4-KV卸载Host-910C尝试-20260903](docs/goals/proto-910c-202609/note_KV卸载Host尝试_910c.md) + probes/910c/kv-offload-host_910c.py |
| 09-01 | **纯 MoE 生成质量退化根因定位（issue #5）**：实锤厂商 `patch_decode_attention`（decode 无条件替换为 prefix-cache prefill_attention）为退化源（非 expert GEMM，dense 同退化），禁用后正常、解码提速近 2x | [新线栈decode生成退化-根因定位-20260901](docs/goals/bringup-p800/note_新线栈decode生成退化根因定位_p800.md) |
| 09-01 | **V3 第一步：KV 卸载到 Host 跑通（P800）**（0.13 官方 OffloadingConnector）：store/load 双向实测、吞吐代价 ~2.4%；num_cpu_blocks=0 接线缺陷已绕过（显式 KVTransferConfig） | [vllm-0.13-allocator与offload调研-20260822](docs/goals/bringup-p800/survey_vllm0.13-allocator与offload调研_p800.md) §4 + probes/p800/kv-offload-host_p800.py, probes/p800/kv-offload-xfer_p800.py |
| 08-22 | 新线镜像纯 MoE 复测：topk_softmax 阻塞绕开（dispatch 降级 reference.torch），expert GEMM 首次触达；eager 可生成但质量退化，默认模式 graph capture 35min 不可用（部分解锁） | [新线镜像纯MoE复测-20260822](docs/goals/bringup-p800/note_新线镜像纯MoE复测_p800.md) |
| 08-22 | P800 V1 显存画像：7 阶段全绿，KV 556,352 tokens/76.40GiB（~89%），910c P0 不存在 | [路线A-P800显存画像报告-20260822](docs/goals/bringup-p800/profile_P800显存画像_p800.md) |
| 08-22 | vllm 0.13 allocator/offload 调研：原生 KV CPU 卸载（--kv-offloading-size）发现 | [vllm-0.13-allocator与offload调研-20260822](docs/goals/bringup-p800/survey_vllm0.13-allocator与offload调研_p800.md) |
| 08-22 | 三 issue 清单成形（causal_conv1d / topk_softmax / 回退开关失效） | [昆仑芯问题反馈清单-20260822](docs/goals/bringup-p800/issue_昆仑芯问题反馈清单_p800.md) |
| 08-22 | 官方镜像 MoE 复测 + 纯 MoE 隔离：定位 causal_conv1d / topk_softmax 双阻塞 | [官方镜像复测-MoE-20260822](docs/goals/bringup-p800/history/note_官方镜像复测MoE-20260822_p800.md) · [纯MoE-昆仑芯-20260822](docs/goals/bringup-p800/history/note_纯MoE隔离测试-20260822_p800.md) |
| 08-21 | 昆仑芯 P800 可用性实测：官方栈端到端跑通 Qwen3-4B（96.5 tok/s），分层验证 + 阻塞项清单 | [路线A-P800可用性实测-20260821](docs/goals/bringup-p800/profile_P800可用性实测_p800.md) |

> 更早条目（2.4 方案定稿、分配器/V1 画像、推理链路闭环、算子三方联调模板、新机器 torch_fl 栈复现等）属已冻结路线，见 [docs/goals/legacy-2.4-910c/README.md](docs/goals/legacy-2.4-910c/README.md)。
