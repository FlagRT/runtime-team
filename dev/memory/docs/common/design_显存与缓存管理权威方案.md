# 显存与缓存管理方案

> 日期：2026-08-22（2026-09-03 更新框架表述）｜ 负责人：xliu969 ｜ 状态：🟢 权威方案
> 适用：昇腾 910c（待验证补充）、昆仑芯 P800（已实测基线）
> 依据：archive/README.md（历史归档索引）、路线A-P800可用性实测-20260821.md、纯MoE-昆仑芯-20260822.md、昆仑芯问题反馈清单-20260822.md
> 早期方案（torch_fl 设备层栈）已冻结归档，见 [archive/README.md](archive/README.md)。

---

## 1. 结论摘要（一屏）

- **技术栈**：FlagOS 官方栈（各芯片厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL）。**显存管理主战场 = vLLM 层**：厂商 torch 自带分配器为底座，分层/溢出逻辑在 vLLM 内做。
- **继承结论**：KV cache 是显存大头（910c：24.3GiB/76% 见 [archive/V1-显存画像报告-20260817.md](archive/V1-显存画像报告-20260817.md)；P800：69.22GiB/504k tokens 见 [路线A-P800可用性实测-20260821.md](路线A-P800可用性实测-20260821.md)）→ 分层缓存/可控溢出的主对象是 KV。
- **基线**：P800 dense 推理已端到端可用（Qwen3-4B 96.5 tok/s，官方发布镜像内）；显存画像数据见 [路线A-P800可用性实测-20260821.md](路线A-P800可用性实测-20260821.md)。
- **阻塞**：MoE 阻塞（混合架构 `xpudnn::causal_conv1d_update ret=1`、纯 MoE `flag_gems.topk_softmax` 4 参签名）影响目标模型选择，需与厂商/智源对接（见 [昆仑芯问题反馈清单-20260822.md](昆仑芯问题反馈清单-20260822.md)）。
- **昇腾侧结论待 910c 机器补充**（见 §7 清单）。

---

## 2. 显存栈结构（按芯片）

| 芯片 | 设备层分配器 | vLLM 版本 | KV 管理 | 显存池状态 |
|---|---|---|---|---|
| **昇腾 910c** | torch_npu（待验证，官方发布配置） | vllm 0.20.2 | vLLM 块表 / prefix caching | torch_npu 自带 caching allocator（**待画像**） |
| **昆仑芯 P800** | xpytorch（CUDA 兼容版，torch.cuda 语义，torch 2.9.0+cu129） | vllm 0.13.0 | vLLM 块表（KunlunxinAttentionBackend，vendor 路径） | torch_plugin 0.1.0 **无独立显存池**（FLAGOS_USE_CACHING_ALLOCATOR/memory_stats 在 P800 缺失）→ 显存池 = torch.cuda caching allocator + vLLM 层管理 |

要点（数据来源 [路线A-P800可用性实测-20260821.md](路线A-P800可用性实测-20260821.md) B.2 / 阻塞项 #6）：
- P800 的 `FLAGOS_USE_CACHING_ALLOCATOR` 开关无意义——厂商栈无独立设备层显存池，V2 A/B 定义必须重做（§4/§5）。
- 昇腾侧 venv 组合须以 torch_npu + vllm 0.20.2 + triton_ascend 3.2.1 组合验证（§7）。

---

## 3. 继承的有效结论（附归档位置）

| 结论 | 内容 | 归档位置 |
|---|---|---|
| **KV 为主对象** | 910c：KV 预分配 170,224 tokens/约 24.3GiB（占加载后 76%）；P800：KV 504,000 tokens/69.22GiB —— 分层缓存/可控溢出的主对象就是 KV | [archive/V1-显存画像报告-20260817.md](archive/V1-显存画像报告-20260817.md)、[archive/显存与缓存管理-2.4-调研与实施方案.md](archive/显存与缓存管理-2.4-调研与实施方案.md) |
| **vLLM KV 块级管理不重复造轮子** | vLLM 自带块表（BlockTable）、prefix caching、chunked prefill；KV 整块分配、块级复用归 vLLM，显存优化围绕它做 | [archive/显存与缓存管理-2.4-调研与实施方案.md](archive/显存与缓存管理-2.4-调研与实施方案.md) §1.2② |
| **vLLM offload 机制** | 0.20.2 offloader 双后端（UVA/prefetch）+ `evict_blocks` 挂载点 + 连接器模式；vllm 0.13 差异调研已完成（2026-08-22）：0.13 原生内置 KV cache CPU 卸载（`--kv-offloading-size`，native/lmcache 后端，`vllm/v1/kv_offload/`）；0.20.2 昇腾侧 native 路径被 `is_cuda_alike` 平台门挡住（见 [routeA-S4-KV卸载Host-910C尝试-20260903.md](routeA-S4-KV卸载Host-910C尝试-20260903.md)）；权重 offload 仅 UVA 单机制（无 offloader 抽象） | [archive/vllm-offload-调研笔记-20260817.md](archive/vllm-offload-调研笔记-20260817.md)、[vllm-0.13-allocator与offload调研-20260822.md](vllm-0.13-allocator与offload调研-20260822.md) |
| **跨组接口问题仍开放** | 编译组执行计划接口（是否含生命周期/复用提示）；算子组 P0 清单（2026-08-16 到期催收） | [archive/执行计划-显存规划-跨组确认-20260817.md](archive/执行计划-显存规划-跨组确认-20260817.md) |

---

## 4. 新问题与工作项

1. **P800 KV 预分配 69.22GiB（gpu_mem_util=0.9 默认，加载后 used 93.80GiB）** → 预分配策略与利用率分析（V1 画像专项，数据见 [路线A-P800可用性实测-20260821.md](路线A-P800可用性实测-20260821.md) S3）。
2. **显存池定义与 V2 A/B 回归重做**：早期开关 `FLAGOS_USE_CACHING_ALLOCATOR` 在 P800 无对应物，A/B 改为 vLLM 层开关 vs 厂商 allocator（依据 [archive/README.md](archive/README.md)）。
3. **vllm 0.13 vs 0.20.2 差异**：offload/evict_blocks/prefix caching/allocator 接口逐项对照（P800 为 0.13.0 昆仑芯构建，昇腾为 0.20.2）。
4. **MoE 阻塞跟踪**（影响目标模型清单确认）：
   - 混合架构：`xpudnn::causal_conv1d_update ret=1`（厂商 SDK，官方发布镜像内默认模式必崩，eager 乱码）——见 [官方镜像复测-MoE-20260822.md](archive/官方镜像复测-MoE-20260822.md)；
   - 纯 MoE（gems 4.2.1rc0）：插件 0.1.0 按 5 参调 `flag_gems.topk_softmax`，kunlunxin 后端 4 参（缺 renormalize），4 组配置全崩——见 [纯MoE-昆仑芯-20260822.md](archive/纯MoE-昆仑芯-20260822.md)；**新线镜像复测（2026-08-22）已绕开该阻塞**：插件 0.2.0 dispatch 多后端降级（vendor.kunlunxin → reference.torch），expert GEMM（厂商 xtorch_ops `_klx_fused_experts`）首次在昆仑芯触达；但 eager 生成质量退化（首 token 正确后重复乱码，#5），默认模式 graph capture ~35min 不可用——见 [新线镜像纯MoE复测-20260822.md](新线镜像纯MoE复测-20260822.md)；新崩溃点 `moe_align_block_size` 6/7 参（#4，新线须用 `VLLM_FL_PREFER=flagos` 单独值规避）；
   - 昇腾对照：topk_softmax 走通用版（5 参）**大概率不受影响**（见 [昆仑芯问题反馈清单-20260822.md](昆仑芯问题反馈清单-20260822.md) 昇腾对照表）；
   - 后续：**复测已完成（结论：部分解锁）**；下一步①定位 eager 生成质量（#5，隔离厂商内核精度/回退路径/输入格式）②graph capture 35min 问题（KunlunxinAttentionBackend 不支持 FULL 模式）③静态预检已确认 gems 5.x 全系（含 v5.3.4）kunlunxin 后端仍未补 renormalize（见 [新线镜像-MoE复测-静态预检-20260822.md](archive/新线镜像-MoE复测-静态预检-20260822.md)）。

---

## 5. 验证计划（V1→V4）

| 步骤 | 内容 | 状态 | 出口 |
|---|---|---|---|
| **V1 P800 显存画像** | 预分配策略、利用率、allocator 统计/碎片、推理峰值曲线（专项报告，另行产出） | ✅ 部分（dense 实测数据已有：加载 84.5s / KV 504k tokens / 69.22GiB / 96.5 tok/s）；专项画像进行中 | 预分配策略与利用率结论 |
| **V1 昇腾画像** | 昇腾（torch_npu + vllm 0.20.2）复测 | ⬜ 910c 待补 | 对照旧栈基线（archive，31.89GiB/24.3GiB KV/76%） |
| **V2 显存池 A/B** | 定义重做（vLLM 层开关 vs 厂商 allocator） | ⬜ 暂缓（依赖 V1 数据） | 开关定义 + 可测吞吐对比 |
| **V3 分层缓存原型** | KV 按需释放 + Host 溢出 | 🔶 部分：**P800（vllm 0.13）KV 卸载到 Host 已跑通**（09-01，官方 OffloadingConnector，store/load 双向实测，吞吐代价 ~2.4%）；**910C（vllm 0.20.2）native 路径不可用**（09-03，平台门 + vllm._C 缺 libcudart，见 [routeA-S4-KV卸载Host-910C尝试-20260903.md](routeA-S4-KV卸载Host-910C尝试-20260903.md)）；待：容量/驱逐/大卸载量实测、昇腾方案定夺 | 容量提升 + 吞吐代价 |
| **V4 SSD 层评估** | NVMe 带宽实测（顺序/随机读写） | ⬜ 随时可做 | SSD 层立项与否 |

---

## 6. 跨组协作（保留接口）

- **编译组**：执行计划接口（是否含张量生命周期/复用提示/峰值估算）——仍开放，继续跟进（[archive/执行计划-显存规划-跨组确认-20260817.md](archive/执行计划-显存规划-跨组确认-20260817.md)）。
- **算子组**：P0 算子清单——影响分层缓存拷贝链路的算子覆盖度，仍在催收。
- **智源/厂商**：三项 issue 提交（按 [昆仑芯问题反馈清单-20260822.md](昆仑芯问题反馈清单-20260822.md)，附 file:line，建议复测结论出来后一并提交）：
  1. `xpudnn::causal_conv1d_update ret=1`（厂商 SDK，混合注意力 MoE 必崩）；
  2. flag_gems kunlunxin 后端 topk_softmax 缺 renormalize（FlagOS 组件，纯 MoE 必崩）；
  3. 文档滞后：官方 install-stack-flagos 技能 vendor-mappings.md 无 kunlunxin 条目、vllm 0.13 与 triton 3.0.0 不匹配（含 platform 插件强依赖未明示）。

---

## 7. 昇腾 910c 待补充清单（另一台机器执行，本机 P800 无法验证）

1. **昇腾 venv 组合验证**：torch_npu + vllm 0.20.2 + triton_ascend 3.2.1。
2. **昇腾 V1 显存画像**：对照旧栈基线（archive，加载 31.89GiB/24.6s、KV 170,224 tokens/24.3GiB/76%、预热 437s）。
3. **昇腾 TP 验证与长序列 prefill P0 复测**：旧栈同场景 2048×4 并发 22min 未完成 prefill（AICore 91-100% 满载，见 archive）须复测（昇腾对照数据见 [路线A-P800可用性实测-20260821.md](路线A-P800可用性实测-20260821.md) S4）。
4. **同条件性能基线**：当前为空白，同卡同模型同 vllm 版本补齐。

---

## 8. 工作原则

- **不预实现**：先有真实问题数据（V1 画像）再动手，按实际 OOM/碎片问题逐步接入，不为方案造轮子。
- **只测不修**：失败保留完整 traceback，不改上游代码；自行变通项单独标注（沿用 [路线A-P800可用性实测-20260821.md](路线A-P800可用性实测-20260821.md) 原则）。
- **测试一律在官方发布镜像内**：dev 容器 triton 版本偏差曾致 FlagGems GEMM 编译崩溃、被误判为组件缺陷；官方发布栈（triton 3.0.0）同 commit 全过——版本偏差会污染结论。
