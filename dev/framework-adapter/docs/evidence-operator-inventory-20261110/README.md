# 清单与ReLU验证证据

- `算子清单-20261110.md`：三模型分阶段逐项表，适合直接阅读。
- `inventory-comparison.json`：实际ATen重载、调用次数、输入形状/精度/stride、模块样本，与固定FlagGems/FlagOP源码注册对应。源码匹配不是环境准入。
- `summary.json`：14轮原始轨迹的离线核验，其中13轮通过，1轮为已保留的检查器失败；最终有效模型实验为原生、重复、候选、关闭、开启、缺审核6组。
- `cpu-controls.log`：7项清单/路由检查器CPU控制。适配包152项CPU控制及wheel身份见本地构建baseline与主报告。

机器原始日志与模型张量保留在本地outputs及本人PPU目录；完整路径/归档SHA在HANDOFF。资格审核及证据原件在case的review-20261110目录，实际启动基线在case/demo。读取结论请先看[主报告](../三模型算子清单与ReLU接入验证-20261110.md)，勿把旧检查器失败解释为数值超差。
