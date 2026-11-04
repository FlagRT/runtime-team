# 框架算子接入与安全回退

本目录属于runtime-team协调仓，维护集成测试、环境与验证材料。功能代码唯一维护在[FlagRT/framework-adapter](https://github.com/FlagRT/framework-adapter)，不在这里复制一套实现。

导航整理：2026-11-03。进度和发布状态只维护于[STATUS](STATUS.md)，不要把历史报告中的“最新/未推送”当作现在。

## 从哪里进入

| 想做什么 | 入口 |
|---|---|
| 看做成什么、缺什么、推了没有 | [当前状态](STATUS.md) |
| 找服务器路径、包和复现步骤 | [运行交接](HANDOFF.md) |
| 看下一阶段的任务与验收 | [本地执行计划](/Users/empty/PhD/Workspace/DomesticRuntime/协作/执行计划.md) |
| 找模型结果、原始证据、历史版本 | [报告索引](docs/README.md) |
| 构建或安装功能包 | [包依赖说明](PACKAGE.md)；在独立功能仓构建 |
| 换agent接手 | [Workspace入口](/Users/empty/PhD/Workspace/DomesticRuntime/README.md) |

## 代码与材料分工

| 目录 | 职责 |
|---|---|
| 独立framework-adapter/src/flagrt_adapter | 策略、记录、注册/恢复、配置、启动、训练入口及模型审核 |
| 独立framework-adapter/probes和tools | 包控制测试、构建与隔离安装验证 |
| 本目录cases/ | 平台baseline、demo、宿主启动器与模型集成测试 |
| 本目录tools/ | [探针与配置入口说明](tools/README.md)，历史兼容启动器 |
| 本目录probes/ | 集成测试与历史实验；qwen_scoped_adapter仅用于09-22复现 |
| 本目录docs/ | 日期报告和精简证据；handoff是跨组材料，archive是历史 |
| 本目录package.lock.json | 已发布功能包依赖，不代表本地dirty版 |

## 现行复现入口

- [P800单算子及三模型](cases/kunlun-unary/README.md)：10-29的限定环境与审核配置。
- [910C旧官方环境](cases/ascend-official-unary/README.md)：成功环境与模型验证。
- [910C训练v3](cases/ascend-train-v3/README.md)：环境检查、原生路径与目标编译失败边界。
- [固定双rank训练配置](demo/training-configs/README.md)：既有训练CLI路径，不能冒充通用框架。

读说明不等于授权启动：上机先确认环境、资源、容器归属和新的结果目录；up.sh --strict会实际起容器。CPU通过不等于硬件通过，命中目标不等于数值通过。

## 开发纲领

先机制后覆盖：只负责接入、准入、选择、恢复和证据，不开发内核或优化模型业务效果。默认保守，原生也须合法；执行后的异常停止，不盲目重试。模型数、芯片数、框架数分开计，完整三级回退按实际缺口验收。

README只做导航；当前结论覆盖更新到STATUS，逐轮实验写日期报告，复现路径放HANDOFF。整理前[方向导航](docs/archive/session-20261103/方向导航整理前-20261103.md)保留，不再按日期向本页追加流水。
