# flagos-runtime（设备上下文统一运行时组件）

> **版本**：v0.1.0 ｜ **状态**：可用（昇腾真机验证通过）｜ **维护**：device-context
> **定位**：运行时层的设备抽象与流语义底座——上承算子层/编译层，下接多机多卡分布式训练推理。
> 上层代码只依赖本组件的统一 API，**换芯片不改代码**（切换后端只改 `use()` 一行）。

---

## 1. 快速接入（下游三行代码）

```python
from runtime import use, set_device, create_stream, translate_error

use("ascend")            # 选择后端（换芯片只改这一行）
set_device(0)            # 绑定设备
s = create_stream()      # 统一流对象
```

## 2. API 面（v0.1.0）

| 域 | 接口 | 说明 |
|---|---|---|
| 后端选择 | `use / available / discover / register` | 注册表机制；新增芯片 = 新增 backend + 跑 conformance |
| 设备 | `device_count / set_device / memory_stats / probe_device` | 枚举、绑定、显存统计、探活 |
| 流 / 事件 | `create_stream / create_event / current_stream / synchronize` | 统一封装；有界同步（超时抛 TimeoutError，防长驻 hang） |
| 错误 | `translate_error / FlagosError / ErrorCategory / DISPOSITION` | 厂商错误码 → L1–L4 分级 + 处置策略（retry/raise/replay/device_recovery） |
| 恢复 | `recover_device / device_state` | 四态监控 + probe/real/hybrid 三级重建 |

完整语义见 **接口约定文档**：`docs/INTERFACE_CONTRACT_DC_20260908.md`。

## 3. 目录结构

```
runtime/
├── __init__.py            # 用户入口（上表 API 由此导出）
├── api/errors.py          # 统一错误对象与分级→处置映射
├── api/stream.py          # Stream / Event 统一封装（含 record_stream 纪律）
├── backends/base.py       # RuntimeBackend 抽象（接口规范，新芯片照此实现）
├── backends/registry.py   # 注册表（register/use/discover）
├── backends/ascend/       # 昇腾后端（torch_npu + 已验证资产复用）
├── conformance/           # 验收用例（13 例 + 6 例）与 runner
├── demos/demo_unified.py  # 设备无关演示
└── smoke_runtime.py       # 接入自检（下游接入后先跑这个）
```

## 4. 质量基线（v0.1.0 交付时状态）

| 项 | 结果 |
|---|---|
| 昇腾真机冒烟 | 37/37 通过（设备/多流/错误分级/恢复） |
| conformance（统一 API 跑历史用例） | 13/13 + 6/6 |
| 跨天长驻 | 28 小时显存零增长（HBM +0.07%） |
| 错误码覆盖 | 108 条映射（CANN 头文件 159 码全集命中 64.8%） |

## 5. 下游接入纪律（两条硬约束）

1. **跨流传缓冲必须 `record_stream`**：多流之间传递张量缓冲时必须调用
   `tensor.record_stream(using_stream)`，否则内存可能被分配器提前回收 → 数据错乱。
   （统一 Stream 已封装 `record_stream` 方法，直接用）
2. **错误处理走分级**：捕获异常后调 `translate_error` 拿 `FlagosError`，
   按其 `disposition` 处置（L1 重试 / L2 上抛 / L3 重放 / L4 走 `recover_device`）；
   **不要**按错误消息字符串自行判断。

## 6. 反馈与迭代

- 问题反馈到 device-context（Kistich），**每周五前**反馈的问题当周定位、下周组件版本修复
- 版本节奏：v0.1 → v0.1.x（吸收 9 月反馈）→ v0.2（10 月，20 模型反馈）
- v0.1 阶段允许接口破坏性变更（会提前知会）；v1.0 起进入稳定接口承诺

## 7. 命名声明

本组件为独立实现，无历史自研路线的命名遗留；vendor 插件目录模式为通用设计。
