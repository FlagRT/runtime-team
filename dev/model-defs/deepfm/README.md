# DeepFM

跨版本共性说明；具体架构/超参/使用方式见各版本目录。

- **场景归属**：传统深度模型场景对齐模型（与生成式 LLM 场景的 `OneRec-8B` 并列），
  见 `docs/运行时层原型验证-战略目标-910C.v1.md` 更新说明（2026-09-21）与
  `dev/model-defs/model_list.md`——推荐项，本轮阶段目标不作硬性约束。
- **上游出处**：架构代码 vendor 自主流开源实现
  [`rixwew/pytorch-fm`](https://github.com/rixwew/pytorch-fm)（MIT License），
  每个版本目录各自保留 `LICENSE`。
- **当前版本**：[`v1/`](v1/)。

## 版本索引

| 版本 | 状态 | 说明 |
|---|---|---|
| [v1](v1/) | 生效 | 标准 DeepFM（wide + FM + deep），随机张量 smoke test 通过 |
