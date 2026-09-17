# 🧊 归档索引（memory 子方向 · 已冻结路线）

> ## ⛔ 红线：本目录内容属于**已冻结路线**，停止开发，不得作为任何新工作的基线或前置依赖
>
> - **冻结对象**：以 **torch_fl 设备层**为底座的整套显存与缓存管理方案（原 2.4 方案 / 昇腾 910C torch_fl 栈画像、推理链路闭环、算子三方联调、新机器 venv 补丁台账等）。
> - **为什么冻结**：2026-08-22 设备层生产主线切换——生产交付统一走 FlagOS 官方栈（各芯片厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL）。torch_fl 设备层路线不承担交付责任、不作前置依赖、不再投入。判据与晋升门槛见 [../../common/policy_设备层路线变更指南.md](../../common/policy_设备层路线变更指南.md)。
> - **接手 memory 子方向请从这里开始**：[../../common/design_显存与缓存管理权威方案.md](../../common/design_显存与缓存管理权威方案.md)（权威方案）+ [../../../PROGRESS.md](../../../PROGRESS.md)（当前进展）。
> - 本目录仅保留**方法论参考**与**历史对照**价值；其中"仍有效结论"列出的数据点须在当前方案下重新验证后方可引用。

---

## 一、冻结清单（torch_fl 设备层栈）

| 文件 | 内容 | 方法论/历史价值 | 当前方案下须重验 |
|---|---|---|---|
| `design_2.4显存与缓存管理方案_910c.md` | 旧实施蓝图（显存池主战场 = torch_fl allocator） | KV cache 是显存大头、分层缓存主对象是 KV；不重复造 vLLM KV 块级管理的轮子 | 显存池主战场改为厂商 torch 自带 allocator + vLLM 管理；V2 显存池回归定义需重做 |
| `profile_allocator画像_910c.md` | torch_fl caching allocator 画像 | 分配器画像方法论可复用 | 当前方案无 torch_fl 显存池，须对厂商 torch 分配器（xpytorch / torch_npu）重新画像 |
| `profile_V1显存画像_910c.md` | 昇腾 910C torch_fl 栈 V1 显存画像 | 加载 31.89GiB、KV 170,224 tokens/24.3GiB(76%)、长序列 prefill 极慢（跨路线问题） | 昇腾侧须以厂商插件组合复测；P800 画像已另行产出 |
| `note_推理插件接入阶段4执行记录_910c.md` | 昇腾 910C torch_fl 栈推理链路闭环 | 单卡→TP 验证流程方法论 | 昇腾链路走厂商官方配置，须重跑 |
| `note_推理插件接入阶段4多卡TP验证_910c.md` | TP 验证（flagcx 通信路径） | TP 数值退化修复经验（浮点归约差异判定） | 昇腾 TP 须重验 |
| `note_算子三方联调测试模板与流程指导_910c.md` | 模板代码在 PyTorch-Plugin-FL 仓 | 三方联调流程方法论（测试归属、模板化验收） | 联调须在 FlagGems 层重建 |
| `note_新机器复现验证_910c.md` | 新机器 torch_fl 栈环境重建 + 分配器画像 5/5 + vLLM 链路组装 | 新机 venv 组装差异记录、flagcx 双卡 allreduce 数值正确 | 属冻结路线复现，不作为当前进展 |
| `note_新机器推理闭环排障第二轮_910c.md` | 新机器 torch_fl 栈推理闭环排障（12 处挂点） | 挂点排查记录 | 属冻结路线复现，不作为当前进展 |
| `patches/`（P1–P4） | 新机器 torch_fl + vllm-plugin-FL venv 补丁台账（flagos_boot / triton npu_utils / vllm base_loader） | 补丁根因与 diff 归档 | 冻结路线专用工装，不在当前交付路径 |

## 二、过程性/重复文档（当前方案，参考保留）

| 文件 | 归档原因 |
|---|---|
| `survey_vllm-offload调研笔记_910c.md` | 内容已并入 `../bringup-p800/survey_vllm0.13-allocator与offload调研_p800.md`（当前方案对照保留） |

> 注：原表中「执行计划-显存规划-跨组确认」「P800适配-执行记录」「官方镜像复测-MoE」「纯MoE-昆仑芯」「新线镜像-MoE复测-静态预检」5 份过程性文档已随本次目录重组移出本目录（分别归入 `../../common/` 与 `../bringup-p800/`、`../bringup-p800/history/`），不再属于本冻结索引，见各自目录的 README。
