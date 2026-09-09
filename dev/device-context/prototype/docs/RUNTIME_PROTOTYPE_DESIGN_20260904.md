# 运行时原型整体设计方案（2026.09 交付）

> **版本**：v0.1（2026-09-04）
> **定位**：对照月度计划 2026.09 节点——「模型转换器原型、基础运行时原型、CI 框架搭建」，
> 考核标准「支持首批 PyTorch/ONNX 模型转换」。
> **思路来源**：参考 torch_fl 的插件式多厂商适配模式，但定位在**运行时 API 层**而非 dispatch 层（见 §2）。
> **关联**：承接 910C 设备上下文已交付资产（D1-D11 全绿、conformance 13+6、错误码映射 108 条、状态恢复真实重建）。

---

## 1. 背景与定位

### 1.1 任务拆解（月度计划 2026.09）

| 交付物 | 说明 | 考核对应 |
|---|---|---|
| **模型转换器原型** | PyTorch / ONNX → 统一 IR | "支持首批 PyTorch/ONNX 模型转换"（硬指标） |
| **基础运行时原型** | 统一 API + 多厂商后端插件骨架 | 运行时原型 |
| **CI 框架** | conformance + 单测流水线 | CI 流水线 |

### 1.2 与 torch_fl 的关系：借鉴模式，不照搬层次

**torch_fl 的模式**（基于 `vllm_fl/dispatch/backends/vendor/{metax,musa,sunrise,thead,txda}` 结构）：
vendor 目录 + dispatch 注册，把各厂商接入 PyTorch 的算子分发体系。

**本原型的关键取舍**：**不做 PyTorch dispatch 层**，做其上一层——**统一运行时 API 层**。理由：

1. 厂商已有原生 PyTorch 扩展（昇腾 `torch_npu`、昆仑芯 `xpupytorch`），**算子分发已被厂商解决**，
   我们再包一层 dispatch 是重复建设；
2. 我们的职责领域正是**设备上下文**（设备/内存/执行句柄、Stream 语义、错误码翻译、状态恢复）——
   这一层厂商**没有统一**，是真正的碎片化所在；
3. 9 月时间盒内，API 层原型可交付、可演示、可复用已有全部资产；dispatch 层原型做不完也演示不好。

**一句话定位**：`厂商负责"PyTorch 能在卡上跑"，我们负责"跑得规范、错得明白、坏能恢复、换卡不改代码"。`

### 1.3 与已有 910C 工作的关系

910C 深度验证（8-9 月）不是绕路，而是**ascend 后端的质保体系**：

| 已有资产 | 在原型中的角色 |
|---|---|
| `errors.py`（108 条映射 + F5 可观测） | ascend 后端的错误码翻译实现 + 全体后端的分级规范 |
| `recovery.py`（R1-R5 + rebuild_mode） | ascend 后端的状态恢复实现 + 恢复接口规范 |
| `device_state.py`（四态机） | 统一 API 的设备状态模型 |
| conformance 13+6 用例 + `--backend` 切换 | 原型的统一 conformance（加一个 backend 就能测一个） |
| 16 项流语义核查 + 事件契约 | Stream/Event 抽象的设计依据与 ascend 质保 |
| 跨天 28h 长驻 / SIGKILL / timeout 真实触发 | 后端成熟度展示材料 |

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────┐
│            用户模型：PyTorch / ONNX                  │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  转换器原型 transform/                                │
│  torch2ir / onnx2ir → 统一 IR（JSON）+ 算子覆盖报告   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  统一运行时 API  runtime/api/                         │
│  设备 · 内存 · Stream/Event · 错误码分级 · 状态恢复    │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Backend 注册表 backends/registry.py                  │
│  runtime.use("ascend") / runtime.use("kunlun")       │
└─────────┬───────────────────────────────┬───────────┘
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│ ascend 后端           │       │ kunlun 后端           │
│ 底座：torch_npu       │       │ 底座：昆仑芯 SDK       │
│ 状态：✅ 实现完整       │       │ 状态：🟨 9月 stub     │
│ 质保：D1-D11 全绿      │       │      11 月插件实做     │
└─────────────────────┘       └─────────────────────┘
          └───────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  CI：conformance --backend {ascend,kunlun} + 单测     │
│  （复用现有 13+6 用例，PR 触发）                        │
└─────────────────────────────────────────────────────┘
```

---

## 3. 核心接口设计（Backend 插件接口规范 v0.1）

插件接口即"规范"——kunlun 9 月照此写 stub，11 月照此做实。**每季度评审一次演进**。

```python
# backends/base.py
from abc import ABC, abstractmethod

class RuntimeBackend(ABC):
    """厂商后端插件接口 v0.1。每个 backend 一个目录，实现全部抽象方法。"""
    name: str            # "ascend" / "kunlun"
    device_type: str     # 用户可见的设备串前缀："npu" / "xpu"

    # ── 设备（职责 D2）──
    @abstractmethod
    def device_count(self) -> int: ...
    @abstractmethod
    def set_device(self, ordinal: int) -> None: ...

    # ── 内存（D3）──
    @abstractmethod
    def memory_stats(self, ordinal: int) -> dict:
        """返回 {"total_mb", "used_mb", "free_mb"}"""

    # ── 执行（D4/D5）──
    @abstractmethod
    def create_stream(self) -> "Stream": ...
    @abstractmethod
    def create_event(self) -> "Event": ...
    @abstractmethod
    def synchronize(self, ordinal: int, timeout_ms: int | None = None) -> None:
        """有界等待；超时抛 TimeoutError（对应 507046 一类）"""

    # ── 错误码翻译（D10）──
    @abstractmethod
    def translate_error(self, exc: BaseException, location: str) -> "FlagosError":
        """厂商错误码 → L1-L4 统一分级，带 mapped/graded_by 可观测字段"""

    # ── 状态恢复（D11）──
    @abstractmethod
    def probe_device(self, ordinal: int) -> bool: ...
    @abstractmethod
    def recover_device(self, ordinal: int, mode: str = "probe") -> bool:
        """mode: probe / real / hybrid（与 recovery.py 语义一致）"""
```

**用户侧入口**（demo 演示的核心）：

```python
import runtime

runtime.use("ascend")            # 或 runtime.use("kunlun") —— 以下代码零改动
dev = runtime.device(0)
s  = runtime.create_stream()
err = runtime.translate_error(exc, location="...")
ok  = runtime.recover_device(0, mode="real")
```

---

## 4. 仓库结构

```
runtime-proto/                       # 9 月新仓库（或 device-context/runtime/ 子目录）
├── runtime/
│   ├── __init__.py                  # runtime.use() / runtime.device() 入口
│   ├── api/
│   │   ├── device.py                # Device 抽象 + 用户 API
│   │   ├── stream.py                # Stream/Event 抽象（依据事件契约 + 16 项流语义）
│   │   ├── memory.py
│   │   └── errors.py                # FlagosError 统一对象（L1-L4 + mapped/graded_by）
│   ├── backends/
│   │   ├── registry.py              # 注册 / 发现 / 分发
│   │   ├── base.py                  # 上面的插件接口规范
│   │   ├── ascend/
│   │   │   ├── backend.py           # torch_npu 实现（收拢已有模块）
│   │   │   ├── error_map.py         # 108 条 ACL 错误码映射（迁入）
│   │   │   └── recovery.py          # 官方重建序列（迁入，含 rebuild_mode）
│   │   └── kunlun/
│   │       ├── backend.py           # 🟨 stub：接口齐全、能力标注 NotImplemented/最小实现
│   │       └── error_map.py         # 昆仑芯错误码占位（11 月填充）
│   ├── transform/
│   │   ├── torch2ir.py              # torch.fx trace → 统一 IR
│   │   ├── onnx2ir.py               # onnx graph → 统一 IR
│   │   └── ir_schema.json           # IR 格式定义（v0.1）
│   └── conformance/                 # 迁入现有 runner + 13+6 用例
├── demos/
│   ├── demo_unified.py              # 同一代码 --backend {ascend,kunlun} 切换
│   └── models/                      # 首批转换模型 3~5 个（resnet18 / bert-tiny / …）
├── .github/workflows/ci.yml
└── docs/
    ├── DESIGN.md                    # 本文档
    └── backend_plugin_spec.md       # 插件接口规范（评审用，从 §3 展开）
```

---

## 5. 后端实现策略

### 5.1 ascend 后端（9 月即完整可用）

- **底座**：直接 import torch_npu——不碰算子分发，只封装设备上下文
- **收拢动作**：把 device-context 里的 `errors.py / recovery.py / device_state.py` 迁入并适配抽象接口
  （逻辑零改动，主要是挪文件 + 对齐方法签名）
- **额外质保**（展示加分项）：ascend 后端是全项目唯一经过深度验证的后端——
  11 项职责全绿、16 项流语义、恢复 8/8、真实重建多进程联调、跨天 28h 长驻零增长

### 5.2 kunlun 后端（9 月 stub，11 月实做）

9 月 stub 的**验收口径**（避免过度投入）：

| 能力 | 9 月 stub 行为 |
|---|---|
| 注册 / 发现 | ✅ 可被 registry 加载，`runtime.use("kunlun")` 成功 |
| device_count / set_device | ✅ 最小实现（读环境或常量） |
| Stream/Event | 🟨 返回未实现异常（`NotImplementedError` 带说明） |
| translate_error | 🟨 返回保守 L3 兜底 + "kunlun 映射表 11 月填充"标注 |
| conformance | 🟨 stub-skip 模式：跑通用例并报告"哪些接口未实现"（**这本身是接口完备度的度量**） |

> stub-skip 报告是 9 月展示的一个巧点：用 conformance 跑 kunlun stub，
> 输出"接口实现度报告"，既证明接口设计完整，又给 11 月实做排出清单。

11 月实做依赖：昆仑芯环境（XPU 卡 + SDK + xpupytorch）——**建议 9 月内提交资源申请**。

---

## 6. 转换器原型设计（统一 IR v0.1）

### 6.1 IR 最小形态（JSON，可读可校验）

```json
{
  "ir_version": "0.1",
  "model": {"name": "bert-tiny", "source": "pytorch", "generated_by": "torch2ir"},
  "graph": {
    "inputs":  [{"name": "input_ids", "dtype": "int64", "shape": [1, 128]}],
    "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 128, 30522]}],
    "nodes": [
      {"id": 0, "op": "Embedding", "inputs": [...], "attrs": {...}},
      {"id": 1, "op": "Matmul",    "inputs": [...], "attrs": {"transposed": false}}
    ]
  },
  "op_coverage": {"total": 24, "matched": 22, "unmatched": ["CustomGatherX"]}
}
```

### 6.2 实现路径

| 路径 | 方法 | 工作量 |
|---|---|---|
| torch2ir | `torch.fx.symbolic_trace` 提取图 → op 名映射到统一算子表 → 输出 IR | 小（fx 现成） |
| onnx2ir | `onnx.load` 遍历 graph.node → 同上映射 | 小 |
| 校验器 | IR schema 校验 + 算子覆盖统计报告（后续对接 10 月算子库的输入） | 小 |

**9 月边界**：只做"**提取 → 映射 → 覆盖报告**"，不做 IR 在后端的执行语义（那是 10 月算子库的事）。
这已满足考核"支持首批 PyTorch/ONNX 模型转换"，且覆盖报告天然是 10 月算子适配的工作清单。

### 6.3 首批模型建议（3~5 个，覆盖代表性算子）

| 模型 | 覆盖算子面 |
|---|---|
| resnet18 | Conv/BN/Pooling/FC（CV 基础面） |
| bert-tiny | Embedding/Matmul/LayerNorm/Softmax（transformer 基础面） |
| 一个 GNN 或推荐小模型（如 DLRM-mini） | Embedding 表 + 交互算子 |
| distilgpt2（可选） | 因果注意力学出 causal mask 语义 |

---

## 7. Conformance 与 CI

### 7.1 conformance（统一验收）

- 迁入现有 runner + 13 训练侧用例 + 6 推理用例
- 扩展 `--backend {ascend, kunlun}`；kunlun 走 stub-skip 模式（§5.2）
- **新后端接入成本 = 实现接口 + 跑 conformance**，这是"接口统一"的直接证明

### 7.2 CI 流水线（.github/workflows/ci.yml 或公司 GitLab CI）

| Job | 内容 | 机器要求 |
|---|---|---|
| unit | 注册表 / IR 转换 / 错误对象单测（CPU 可跑） | 普通 runner |
| conf-ascend | conformance --backend ascend | **自托管 runner（910C）** |
| conf-kunlun | conformance --backend kunlun（stub-skip 报告） | 普通 runner |

PR 触发，徽章挂 README。910C 自托管 runner 需要 27 宿主机装一个 actions runner（待与团队确认 CI 平台）。

---

## 8. 9 月实施节奏（按周）

| 周 | 内容 | 产出 |
|---|---|---|
| W1（~9/5） | 插件接口规范 v0.1 定稿 + 仓库骨架 + registry | `base.py` + `registry.py` + 骨架 |
| W2（9/8-9/12） | ascend 后端收拢（errors/recovery/device_state 迁入）+ `demo_unified.py --backend ascend` 跑通 | ascend 后端可用 |
| W3（9/15-9/19） | 转换器原型（torch2ir + onnx2ir）+ 首批 3~5 模型转换 + 覆盖报告 | 转换器原型（**考核硬指标**） |
| W4（9/22-9/26） | kunlun stub + CI 流水线 + 双后端 conformance + 交付材料（架构图/接口规范/转换报告） | 三项交付齐 + 演示 |

> 注：W1 与昆仑芯调研（另一条线）并行，不冲突。

---

## 9. 风险与开放问题

| # | 风险/问题 | 影响 | 对策 |
|---|---|---|---|
| R1 | **统一 IR 的边界预期**：9 月版本只做图提取+算子映射+覆盖报告，不做执行 | 评审时可能被问"转换完能跑吗" | 文档明确 9 月/10 月边界；覆盖报告即 10 月算子库工作清单，转化为正向叙事 |
| R2 | 昆仑芯环境未到位 | 11 月插件实做阻塞 | 9 月内提交资源申请；stub 先行不阻塞 9 月 |
| R3 | CI 自托管 runner 需要 910C 常驻 agent | 与现有任务（训练/serve）抢卡 | conf-ascend job 限定空闲卡（ASCEND_RT_VISIBLE_DEVICES）；与团队确认 CI 平台 |
| R4 | 与 torch_fl 的边界 | 上层可能问"和 torch_fl 什么关系" | 定位话术：torch_fl 解决"PyTorch 算子分发到多芯片"，本原型解决"设备上下文统一"（错误/恢复/流语义），**层次互补**；后续可在 torch_fl vendor 目录里挂我们的 backend |
| R5 | 接口规范过早固化 | 11 月 kunlun 实做时发现接口不合理 | 接口版本化 v0.1，标注"原型期，允许破坏性变更"；每季度评审 |

---

## 10. 已有资产 → 原型模块映射（迁移清单）

| 现有资产（device-context） | 去向 | 动作 |
|---|---|---|
| `conformance/errors.py`（108 条映射 + F5） | `backends/ascend/error_map.py` | 迁移 + 适配接口 |
| `conformance/recovery.py`（R1-R5 + rebuild_mode） | `backends/ascend/recovery.py` | 迁移 + 适配接口 |
| `conformance/device_state.py`（四态） | `runtime/api/device.py` 状态模型 | 迁移 |
| `conformance/runner.py` + 13+6 用例 | `runtime/conformance/` | 迁移 + 加 kunlun stub-skip |
| `docs/event_semantics_contract.md` | Stream/Event 抽象设计依据 | 引用 |
| 16 项流语义核查（§5.4） | Stream 抽象的测试基线 | 引用 |
| 双缓冲 v2（四模式选型） | 后续 `runtime/pipeline/`（10 月+） | 暂不动 |

**原则：迁移为主、新写为辅。** 9 月新写的代码 = 抽象基类 + registry + 转换器 + kunlun stub + CI 配置。
