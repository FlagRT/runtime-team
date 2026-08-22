# 归档索引（memory 子方向 · 2026-08-22）

- 日期：2026-08-22
- 范围：`dev/memory/docs/` 下 B 线旧案收拢
- 依据：《FlagOS设备层路线变更指南.md》

---

## 一、归档原因

2026-08-22 完成设备层路线变更：生产主线由 B（torch_fl 设备层）切换为 A（各芯片厂商设备插件 + FlagGems + FlagCX + vllm-plugin-FL，即官方发布配置）。memory 子方向旧案（2.4 显存与缓存管理）整套调研、画像与执行记录均构建在 B 的 torch_fl 显存池 / 昇腾 910c B 栈之上，不进入 A 线交付路径，故收拢为本归档，供方法论参考与 A 线重验对照。

## 二、归档清单（B 线旧案）

| 文件名 | 原位置 | 废弃理由 | 仍有效结论摘要 | 需 A 线重验 |
|---|---|---|---|---|
| 显存与缓存管理-2.4-调研与实施方案.md | dev/memory/docs/ | B 线实施蓝图（显存池主战场 torch_fl allocator） | KV cache 是显存大头（910c 实测 24.3GiB/76%），分层缓存/可控溢出的主对象是 KV；vLLM 自带 KV 块级管理不重复造轮子；跨组接口问题（编译组执行计划、算子组 P0 清单）仍开放 | 显存池主战场改为厂商 torch 自带 allocator + vLLM 管理（P800 上 torch_plugin 0.1.0 无 torch_fl 显存池）；V2 显存池 A/B 回归定义需重做 |
| allocator-画像报告-20260817.md | dev/memory/docs/ | torch_fl caching allocator 画像（B 线资产） | 探针方法论（分配器画像方式）可复用；torch_fl caching allocator 能力（块复用/合并/碎片 4.9%）作为 B 线资产记录 | A 线无 torch_fl 显存池，需对厂商 torch 分配器（xpytorch/torch_npu）重新画像 |
| V1-显存画像报告-20260817.md | dev/memory/docs/ | 昇腾 910c B 栈 V1 显存画像 | 910c 昇腾加载 31.89GiB、KV 预分配 170,224 tokens/24.3GiB（76%）；P0 长序列 prefill 极慢（跨路线问题） | 昇腾 A 线（torch_npu+vllm 0.20.2）复测；P800 A 线画像已另行产出（路线A-P800可用性实测-20260821.md 含加载 84.5s/KV 504k tokens/69.22GiB） |
| 推理插件接入-阶段4执行记录.md | dev/memory/docs/ | B 线昇腾推理链路闭环（torch_fl 设备层） | 昇腾 910c B 线推理链路闭环方法论（单卡→TP 验证流程） | A 线昇腾链路走 torch_npu 官方配置，链路验证需重跑 |
| 推理插件接入-阶段4多卡TP验证-执行记录.md | dev/memory/docs/ | B 线 TP 验证（flagcx 通信路径） | TP 数值退化修复经验（浮点归约差异判定） | A 线昇腾 TP 验证 |
| 算子三方联调-测试模板与流程指导-20260816.md | dev/memory/docs/ | 模板代码在 PyTorch-Plugin-FL 仓（B 仓） | 三方联调流程方法论（测试归属、模板化验收） | 模板代码在 B 仓，A 线联调需在 FlagGems 层重建 |

## 三、保留清单（A 线沿用）

| 文件名 | 保留原因 |
|---|---|
| vllm-offload-调研笔记-20260817.md | vLLM 0.20.2 分层/溢出机制（offloader 双后端 + evict_blocks 挂载点）是 vLLM 特性，与设备层路线无关；注意 P800 用 vllm 0.13，需对照（vllm-0.13 调研待补） |
| 执行计划-显存规划-跨组确认-20260817.md | 编译组执行计划接口问题仍开放，A 线继续跟进 |
| P800适配-执行记录-20260820.md | P800 环境验证历史（A 线环境基线：驱动/容器/xpytorch/flag_gems vendor 识别），保留作 A 线环境参考 |

---

> 新方向权威方案见 [../路线A-显存与缓存管理-方案-20260822.md](../路线A-显存与缓存管理-方案-20260822.md)（另行发布）。
