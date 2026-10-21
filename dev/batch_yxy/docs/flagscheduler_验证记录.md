# FlagScheduler 0.1.0 验证记录

日期：2026-10-21。以下是本轮实际执行结果，不表示完整生产或真机模型验收。

| 检查项 | 实际结果 | 范围 |
| --- | --- | --- |
| 原有测试 | 117 项通过 | 旧动态组批、单次调度、结果存储、设备/模型契约 |
| 全部测试 | **171 passed** | 包含 CLI/配置、常驻 runtime、回环 HTTP、输入边界、失败回填、ID 重排映射、背压、退出排空、并发接入、15 类模型 mock、启动计划保护，以及新增脚本目录/快照/选择测试 |
| 测试运行方式 | CPU 隔离容器，根文件系统只读，无网络/GPU | 仅项目目录可写，已有 pytest/numpy 依赖只读挂载，临时文件位于 `.tmp/` |
| 包构建 | `flagscheduler-0.1.0-py3-none-any.whl` 构建成功 | 无安装到外部环境；产物位于 `artifacts/dist/` |
| 包冒烟 | 从 wheel 直接导入，关闭 site-packages，console entry point 校验通过 | 无第三方依赖的 Mock 三请求完成 |
| CLI 生命周期 | `serve → status → submit → SIGTERM` 通过 | CPU 容器内项目只读挂载，以普通用户运行，完成请求并正常排空退出 |
| 框架发现 | 找到 Conda 多环境 | `vllm` 环境 0.21.0，另有 0.10.0 / 0.29.0+precompiled 的 vLLM 环境、0.5.9 SGLang 及多个 Transformers 环境；不混用版本 |
| Scheduler 只读定位 | 找到 0.21.0 同步/异步 V1 scheduler 与扩展字段 | 记录源码路径与指纹；不读取运行进程的当前 scheduler |
| 同步插件导入 | `FlagScheduler` 校验通过 | 与真实 0.21.0 的 SchedulerInterface 继承/非抽象/构造签名校验，不加载模型 |
| 异步插件导入 | `FlagAsyncScheduler` 校验通过 | 同上；继承原生异步行为 |
| 静态启动计划 | 已生成 `artifacts/vllm-0.21.0-plan.json` | 模型路径仍为占位值，含自定义和原生 argv；未执行启动 |
| Default/Currently | 初始默认双标记、选择项目脚本、回切 Default、按持久选择生成计划均通过 | 在实际 Conda vLLM 0.21.0 环境只读检查，运行结束将本地选择恢复为 Default |
| 原生脚本保留 | 原文件指纹前后一致，所有已发现原生脚本留有只读 SHA-256 快照 | `.state/schedulers/originals/`，不覆盖原生文件或已有快照；同步/异步、环境别名与版本隔离均有测试 |
| NVIDIA 清单 | 4 × RTX A6000，驱动 595.71.05 | 只读设备清单，不代表空闲或完成了推理验证 |
| 文件范围 | Git 改动全部位于 `dev/batch_yxy/` | 未更改仓库根忽略规则、其他子目录或外部 Conda 源码 |

当前没有加载真实模型、替换运行中服务、运行 GPU 推理或测量设备性能。提供的是可执行的节点骨架与真实 vLLM 接口接入能力；示例内部 scheduler 仍继承原生行为。

下一项设备验收需要明确本地权重路径/版本、指定空闲 GPU，并按使用指南执行原路径/统一入口及原生/自定义 scheduler 的数值、请求映射、预算、退出和性能对照。TUI、其他真实后端、SLO 算法及生产级资源隔离依照架构路线后续实现。
