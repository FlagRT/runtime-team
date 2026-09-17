# ✅ proto-910c-202609 —— 运行时层原型验证（910C，本期已交付）

> 目标：锁定推理镜像 `vllm-ascend:v0.20.2rc1-a3` + 验收模型 `Qwen/Qwen3-Embedding-0.6B`，对应战略文档「运行时层原型验证-战略目标-910C」§3/§5/§7 的本期验收范围（显存池定义 + 显存占用峰值画像）。
> 状态：✅ 已交付（2026-09-10/11）。

| 文件 | 内容 |
|---|---|
| `profile_显存画像_910c.md` | 910C 锁定镜像显存画像报告：加载阶段结构分解 + 运行阶段峰值 sweep（batch×seqlen）+ A/B 三轴对照 |
| `design_显存池定义_910c.md` | 显存池定义：torch_npu caching allocator 底座 + vLLM 层（gpu_memory_utilization/pooling 预分配/ACLGraph capture），含安全区间建议 |
| `note_KV卸载Host尝试_910c.md` | S4 KV 卸载到 Host 的 910C 移植尝试记录：阻塞留档（平台门 + vllm._C 缺 libcudart） |

跨方向速览与最新结论见根 [STATUS.md](../../../STATUS.md)。
