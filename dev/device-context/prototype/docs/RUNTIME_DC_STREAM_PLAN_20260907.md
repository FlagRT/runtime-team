# 运行时层 · 设备上下文与多流 Stream
## 职责框架（聚焦版）+ 本周工作计划（2026-09-07 ~ 09-11）

> **版本**：v1.0（2026-09-07）
> **定位**：运行时原型（9 月月度节点）中，**我们负责的子层**——设备上下文 + 多流 Stream。
> **架构位置**：上承算子层 / 编译层（与模型转换器），下接多机多卡分布式训练与推理。
> **工作方式**：由我们**起头搭建运行时层的整体框架**（插件机制 + 接口规范），
> 但**只实现自身职责范围内的模块**；其余子域留给对应方向。

---

## 1. 职责边界（明确"做"与"不做"）

| 层 | 模块 | 归属 | 我们是否实现 |
|---|---|---|---|
| 上层 | 模型转换器（PyTorch/ONNX → 统一 IR） | 模型/编译方向 | ❌ 不实现（我们提供接口约定） |
| 上层 | 算子库 / 算子适配 | 算子方向（10 月） | ❌ 不实现 |
| **本层** | **Backend 插件机制 + 注册表** | **我们（起头）** | ✅ |
| **本层** | **设备上下文（设备/内存/执行句柄）** | **我们** | ✅ |
| **本层** | **多流 Stream（流/事件/同步语义）** | **我们** | ✅ |
| **本层** | **错误码翻译与分级** | **我们**（已有 108 条映射） | ✅ |
| **本层** | **设备状态恢复** | **我们**（已有 R1-R5 + 真实重建） | ✅ |
| 本层 | 通信（集合通信适配） | 通信方向（FlagCX） | ❌ 不实现 |
| 下层 | 多机多卡分布式训练/推理 | 训练/推理方向 | ❌ 不实现（我们是其底座） |

**边界话术**：我们交付的是**运行时层的"设备抽象与流语义"底座**——上层算子/编译/转换器按我们的接口接入，
下层分布式训推站在我们的抽象之上。"换卡不改代码"由我们的抽象保证。

---

## 2. 910C 深度验证分支：收口确认 ✅

深度下钻阶段（8-9 月）已达成目标，可作为**ascend 后端的质保体系**收口：

| 交付 | 状态 |
|---|---|
| 11 项职责（训练侧 + 推理侧双侧） | ✅ 全绿，PR #11 已于 9-02 合入 dev-1.0（157 文件） |
| 多流 Stream 16 项子项核查 | ✅ 15 通过 / 1 不适用 |
| 错误码映射 | ✅ 108 条 + F5 可观测（mapped / graded_by） |
| 状态恢复 | ✅ R1-R5 + 真实重建（CANN 官方序列）+ 多进程联调（隔离性成立） |
| 集成（错误翻译 + 状态监控挂真实 vLLM） | ✅ 实测 VLLMValidationError → L2_PARAM |
| 长驻 | ✅ 跨天 28 小时 HBM +42MB（+0.07%）零增长 |
| 回归 | ✅ conformance 13/13 + infer 6/6（多轮） |

**随框架延续的 4 项**（均需外部环境，不阻塞收口，挂到新框架继续）：
D11 real 多卡多进程压测 · 芯片级错误真实触发 · 流优先级调度效果 · 训练侧完整 epoch 吞吐复测。

---

## 3. 运行时层整体框架（我们起头搭的骨架）

```
            算子层 / 编译层 / 模型转换器  ── 其他方向
                        │  调用统一接口
┌───────────────────────▼────────────────────────┐
│  统一运行时 API（runtime/api/）                 │
│  ┌──────────┬──────────┬─────────┬───────────┐ │
│  │ 设备上下文 │ 多流Stream│ 错误码   │ 状态恢复  │ │  ← 我们实现
│  └──────────┴──────────┴─────────┴───────────┘ │
└───────────────────────┬────────────────────────┘
                        ▼
        Backend 注册表（backends/registry.py）      ← 我们起头
              ├── ascend/（torch_npu）  ✅ 本周落地
              └── kunlun/（昆仑芯 SDK） 🟨 stub（W4）
                        │
        分布式训练 / 分布式推理（下层使用者）
```

**框架的核心价值**：新增一家芯片 = 实现一个 backend + 跑通 conformance。
这是"统一接口"最直接的证明，也是我们留给团队最重要的资产。

---

## 4. 设备上下文接口规范（聚焦版）

```python
class DeviceContextBackend(ABC):
    name: str;  device_type: str          # "ascend"/"npu"、"kunlun"/"xpu"

    # ── 设备句柄与生命周期 ──
    @abstractmethod
    def device_count(self) -> int
    @abstractmethod
    def set_device(self, ordinal: int) -> None
    @abstractmethod
    def memory_stats(self, ordinal: int) -> dict     # total/used/free（MB）

    # ── 执行句柄 ──
    @abstractmethod
    def current_stream(self) -> "Stream"
    @abstractmethod
    def synchronize(self, ordinal: int, timeout_ms: int | None = None) -> None

    # ── 错误码翻译 ──
    @abstractmethod
    def translate_error(self, exc, location: str) -> "FlagosError"
    #   → L1 重试 / L2 上抛 / L3 重放 / L4 设备恢复；带 mapped + graded_by

    # ── 状态恢复 ──
    @abstractmethod
    def probe_device(self, ordinal: int) -> bool
    @abstractmethod
    def recover_device(self, ordinal: int, mode: str = "probe") -> bool
    #   mode: probe（默认保底）/ real（真实重建）/ hybrid（先探针后重建）
```

**已有实现直接迁入**：`errors.py`（108 条）→ ascend/error_map.py；`recovery.py`（R1-R5 + rebuild_mode）
→ ascend/recovery.py；`device_state.py`（四态）→ api 状态模型。**逻辑零改动，仅对齐签名。**

---

## 5. 多流 Stream 接口规范（聚焦版）

```python
class Stream(ABC):
    def wait_event(self, ev: "Event") -> None        # 跨流依赖
    def wait_stream(self, other: "Stream") -> None   # 粗粒度等待
    def synchronize(self, timeout_ms: int | None = None) -> None

class Event(ABC):
    def record(self, stream: "Stream" = None) -> None
    def query(self) -> bool          # 未 record 时语义须明确（已有契约）
    def wait_host(self, timeout_ms: int) -> bool     # 有界等待
```

**验收基线 = 已完成核查的 16 项子项**（新后端接入时逐项比对）：

| 域 | 子项 |
|---|---|
| 基础语义 | 流内顺序 · 显式依赖（无隐式同步）· 跨流可见性 · wait_stream 传递 |
| 并发与绑定 | 多流并发重叠 · 集合通信与流绑定（须有排他证据）· 图捕获流语义 |
| 工程安全 | 默认流 vs 命名流 + `record_stream` · 错误隔离分层 · 流/事件生命周期与配额 · 跨流内存分配 |
| 能力与限制 | 流优先级 · 多设备流绑定 · 同步超时 · 跨进程共享（IPC，标注不适用）· 流数量配额 |

**两条硬性纪律（写入接口文档）**：
1. 跨流传递内存必须 `record_stream(using_stream)`，否则分配器可能提前回收 → 数据竞争。
2. 错误隔离分层：API 级失败只影响该次调用；芯片级故障影响该设备全部流，
   恢复须走**设备级** `recover_device(mode="real")`，流级重试无效。

---

## 6. 本周工作计划（2026-09-07 ~ 09-11，W2）

| # | 任务 | 产出 | 验收标准 | 天数 |
|---|---|---|---|---|
| 1 | 仓库骨架落地 | `runtime/` 包：`api/`、`backends/base.py`、`backends/registry.py` | `runtime.use("ascend")` 成功加载 | 0.5 |
| 2 | 接口规范 v0.1 代码化 | `base.py` 抽象基类（设备上下文 + Stream 两域，见 §4/§5） | 抽象接口完整、可被实现 | 0.5 |
| 3 | **ascend 后端收拢** | `backends/ascend/{backend,error_map,recovery}.py`（迁入 errors / recovery / device_state） | 逻辑零改动，签名对齐；单测可 import | 1.5 |
| 4 | **多流 Stream 抽象** | `runtime/api/stream.py`（Stream / Event 抽象，依据事件契约 + 16 项子项） | 抽象覆盖 16 项子项语义 | 1 |
| 5 | conformance 接入 | 迁入 runner + 13/6 用例，扩展 `--backend ascend` | **13/13 + 6/6 PASS**（与 910C 阶段一致） | 1 |
| 6 | demo 跑通 | `demos/demo_unified.py --backend ascend` | 910C 上跑通，展示设备无关调用 | 0.5 |

**本周交付物**：可 import、可运行的 `runtime/` 骨架 + ascend 后端 + 13/6 conformance 全绿 + 一个 demo。

**本周不做**（避免范围蔓延）：kunlun stub（W4）、转换器（其他方向 / W3）、CI 流水线（W4）。

---

## 7. 与其他方向的接口约定

| 对接方 | 约定 |
|---|---|
| 模型/编译方向（转换器） | 我们提供 `runtime.use()` + 设备抽象；转换器只产出 IR，不感知芯片 |
| 算子方向（10 月） | 算子经厂商 torch 扩展落地；我们的 Stream/Event 抽象供其调度使用 |
| 通信方向（FlagCX） | 我们不实现通信；但集合通信须遵守"与流绑定"语义（我们提供 F 场景排他验证方法） |
| 训练/推理方向（下层） | 站在我们抽象之上；换卡只改 `runtime.use()` 一行 |

---

## 8. 风险与提示

| 项 | 说明 | 对策 |
|---|---|---|
| 接口过早固化 | kunlun 11 月实做可能发现接口不合理 | 标注 v0.1"原型期允许破坏性变更"，每季度评审 |
| ascend 收拢引入回归 | 迁移过程可能改动行为 | **先跑 conformance 13/6 建立基线，再迁移，迁移后再跑**（前后对比） |
| 仓库位置未定 | 独立新仓库 vs device-context 子目录 | 本周先用 `device-context/runtime/` 子目录推进，位置后续可整体搬迁（成本极低） |
| 昆仑芯环境 | 11 月实做需要 | 9 月内提交资源申请（本周可先起流程） |
