# 统一设备抽象封装 · 路线 A 与路线 B 总结

> 版本：v1.0（2026-09-22）｜ 负责人：Kistich（hliu553）
> 方向：运行时层 · 设备抽象与执行上下文 + 多流 Stream
> 芯片口径：华为昇腾 910C（已完成）· 昆仑芯 P800（已完成）· 寒武纪 MLU590（已完成）；平头哥 PPU 备选
> **路线 B（torch_fl）于 2026-09-22 整体退出**：原型里的该后端已删除，三家实例当下与今后一律走路线 A。

---

## 结论（一句话）

**我们选用路线 A —— 厂商官方 torch 插件基线，不走路线 B —— torch_fl。**
这不是偏好选择，而是三条硬约束推出来的唯一可行解：**torch_fl 覆盖不全（没有寒武纪、没有壁仞）、
与厂商栈争抢同一个 PyTorch 接入位（同进程互斥）、且性能差 2.3 倍**（同机实测 5428 vs 2324 tok/s）。
我们的价值在厂商栈**之上**的统一抽象层，而不在替换厂商栈。

---

## 一、为什么是路线 A 而不是路线 B

### 1.1 两条路线是什么

**路线 A = 厂商官方 PyTorch 设备插件基线。** 每家芯片使用其厂商官方维护的 PyTorch 设备插件，
我们**不替换、不改写**厂商插件，只在其之上做统一封装。

| 芯片 | 厂商官方 PyTorch 栈 | 设备串 | 底层运行时 / 通信 |
|---|---|---|---|
| 华为昇腾 910C | `torch_npu` | `npu` | CANN 9.0 · HCCL |
| 昆仑芯 P800 | `torch.cuda`（XPytorch + torch_xray 符号重写） | `cuda` | BKCL · XCCL |
| 寒武纪 | `torch_mlu`（Torch-MLU，已开源） | `mlu` | NeuWare / CNRT · CNCL |
| *（备选）平头哥 PPU* | *T-Head SAIL* | *待定* | *厂商自研软件栈* |

**路线 B = torch_fl**，即 FlagOS 侧的统一加速器抽象框架。

### 1.2 关于路线 B 的调研结论（为什么用不了）

| # | 问题 | 调研依据 |
|---|---|---|
| 1 | **覆盖不全，"通吃"前提不成立** | torch_fl 平台矩阵与 CI 配置实有 9 家：`cuda / metax / ascend / ppu / dcu / gcu / musa / bpu / tsingmicro`。其中** `ppu`（平头哥）状态为 Experimental**；**没有 cambricon（寒武纪）、没有 biren（壁仞）**。走 torch_fl这条路适配成本过高 |
| 2 | **与厂商栈争抢同一个接入位（硬约束）** | torch_fl（`flagos`）与 `torch_npu`（`npu`）、`torch_mlu`（`mlu`）**都占 PyTorch PrivateUse1**，而同进程只能激活一个。昇腾实测：锁定训练镜像**禁止 torch_npu 与 torch_fl 共存**，必须 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 且先 `import torch_fl` 再 `import torch` |
| 3 | **性能代价明确且很大** | 同机、同模型、同卡数实测：`torch_npu` + 原生 HCCL **5428 tok/s** vs torch_fl **2324 tok/s**，**差 2.3 倍** |
| 4 | **生态默认适配对象是厂商栈** | 上层框架（vLLM / DeepSpeed / FlagGems）与厂商栈的适配由厂商持续推进，我们跟随即可 |

> **一句话**：厂商官方栈是**长期维护、性能最优、生态默认适配**的对象；
> 我们的价值在于它之上的统一抽象层，而不在于替换它。

---

## 二、路线 A 具体怎么走（设计方案）

### 2.1 覆盖范围：五域

```text
设备上下文  →  设备枚举 / 绑定 / 显存口径（total · used · free）
多流 Stream →  Stream·Event 语义（跨流依赖 / 有界等待 / 上下文切换 / record_stream）
错误翻译    →  厂商错误 → L1–L4 统一分级 + 处置策略（retry / raise / replay / device_recovery）
状态恢复    →  四态机 + 五段式恢复（probe / real / hybrid）
插件机制    →  Backend 抽象 + 注册表 + 能力声明 + 已知问题清单
```

### 2.2 接口契约：13 个抽象方法

```text
设备 4 : device_count / set_device / memory_stats / current_stream
多流 7 : create_stream / create_event / synchronize / stream_context
         synchronize_stream / wait_event_host /（Stream·Event 统一包装）
错误 1 : translate_error
恢复 1 : probe_device / recover_device
```

**新增一家芯片的完整动作**（这是"屏蔽芯片差异"的可验收形态）：

```text
新建 backends/<vendor>/    →  实现上述 13 个抽象
        →  build() 工厂登记
        →  supports() 如实声明能力（不支持就不声明，禁止伪造）
        →  known_issues() 如实登记上游缺陷与规避
        →  conformance 13 例 + 推理 6 例全绿（或如实 stub-skip 并说明缺口）
```

### 2.3 关键设计点

| 设计点 | 做法 | 目的 |
|---|---|---|
| **一行切换** | 用户代码只改 `runtime.use(name)` | "同一份代码换芯片"可验收 |
| **能力如实声明** | `supports(capability)` + `_capabilities`，不支持项**不伪造** | 上层可预判，conformance 可判 |
| **统一对象** | `Stream` / `Event` / `FlagosError` 三类统一包装 | 上层不感知厂商对象类型 |
| **可观测字段** | `mapped` / `graded_by` / `is_grade_confident`；恢复返回统一 dict | 上层不把"兜底分级"当确定结论 |
| **缺陷透明** | `known_issues()` 结构化登记（条件 / 症状 / 根因层 / 规避 / 复现率 / 证据） | 接入方读到后端即知坑 |
| **进程级隔离** | 同一进程激活一家；跨进程可切 | 呼应 PrivateUse1 单例的客观约束 |

### 2.4 分层与统一调用链

```text
上层：训练脚本站 / 推理服务 / 算子 / 编译
        │  只调用 runtime.*（唯一与芯片相关的一行：runtime.use("...")）
        ▼
【本方向】统一运行时 API  →  Backend 插件注册表
        │
        ├── runtime.use("ascend")     ──┐
        ├── runtime.use("kunlun")     ──┤  进程内激活一家
        └── runtime.use("cambricon")  ──┘
        ▼
torch_npu  │  torch.cuda(XPytorch)  │  torch_mlu  │  (备选) SAIL
        ▼
CANN/HCCL  │  BKCL/XCCL  │  NeuWare/CNRT/CNCL
```

```text
runtime.use("cambricon")                → 取到 cambricon backend 单例
runtime.device_count()                  → torch.mlu.device_count()
runtime.set_device(0)                   → torch.mlu.set_device(0)
s = runtime.create_stream()             → Stream(统一包装) ⇄ torch.mlu.Stream()
   s.wait_event(ev)                      →   跨流依赖（语义统一）
   s.synchronize(timeout_ms=3000)         →   有界等待（不支持时如实声明并降级）
runtime.memory_stats(0)                 → {"total_mb","used_mb","free_mb"}
runtime.translate_error(e, "op:matmul") → FlagosError[L3_EXECUTION] disposition=replay
runtime.recover_device(0, mode="real")  → {"recovered": true, "detail": "..."}
```

## 三、路线 A 目前的实现情况

### 3.1 概览

```text
第 1 家  华为昇腾 910C   ✅ 已完成（两条腿全闭环 + 长稳 + 按统一判据复跑）
第 2 家  昆仑芯 P800     ✅ 已完成（阶段 0–5 全部完成 + 镜像等价性 + 对称复跑）
第 3 家  寒武纪 MLU590   ✅ 已完成（接入 + 多流 16 项 + 训练腿 + 错误闭环全绿）
备选     平头哥 PPU      ⏳ 待命
```

### 3.2 第 1 家 · 华为昇腾 910C（已完成）

| 项 | 结果 |
|---|---|
| 统一 API + Backend 插件机制 + 注册表 | ✅ 交付 |
| conformance | ✅ 13/13 + 推理 6/6（`ascend`） |
| 冒烟自检（按统一判据复跑） | ✅ **51 通过 / 0 失败** |
| 多流 Stream 16 项子项核查 | ✅ 15 通过 / 1 不适用 |
| 错误码翻译 | ✅ 108 条 ACL 映射 + 可观测字段（`mapped` / `graded_by`） |
| 状态恢复 | ✅ 四态机 + 五段式（R1–R5）+ 真实重建，多进程隔离性验证通过 |
| 训练腿（2 卡分布式微调） | ✅ 6/6 · loss **15.4498 → 11.1479**（50 步）· **3954–4402 tok/s** · `dist=hccl` · 通信三类对照全对 |
| 推理腿（单卡前向 / 服务化） | ✅ 前向 **14/14**（79.4 句/s · p50 37.7 ms · 区分度 0.638）；服务化 10/10（**108 句/s** · p50 27.4 ms） |
| 错误注入 → 恢复闭环 | ✅ 推理腿 5 闭环 / 0 失败；训练腿 **5 闭环 / 0 跳过 / 0 失败** |
| 长稳 | ✅ 跨天 **28 小时** HBM 零增长（+0.07%） |

### 3.3 第 2 家 · 昆仑芯 P800（已完成）

| 项 | 状态 | 结果 |
|---|---|---|
| 阶段 0–1 环境 + 接入 | ✅ | `kunlun` backend 落地；**单日完成接入**）；conformance **13/13 + 6/6**；smoke **42 通过 / 0 失败** |
| 阶段 2 训练腿 | ✅ | 两 rank 6/6 · loss **15.4488 → 11.1481** · **3482 tok/s** · 通信三类对照全对 |
| 阶段 3 推理腿（前向 + 服务化） | ✅ | 前向 **13/13**（维度 1024 · 区分度 0.6392 · 53.1 句/s · p50 56.2 ms）；服务化 **10/10** |
| 阶段 4 错误闭环 | ✅ | 设 / 不设 `XPU_EVENT_KL3_ENABLE` **两组各 5 闭环 / 0 失败，且逐字节一致** |
| 阶段 5 收敛（三件套） | ✅ | 《新芯片接入手册》+ 接口约定修订建议 6 条 + 原型 release `runtime-v0.2.0` |
| 多流 16 项基线逐项比对 | ✅ | **14 通过 / 1 如实标注不支持（S-12 流优先级）/ 1 不适用**；与第 1 家**仅 1 项差异** |
| 官方推荐镜像等价性 | ✅ | 结论全部复现（推理腿 `detail` 逐字相同、厂商缺陷一致重现）⇒ 推荐以官方 `-base` 入锁 |

### 3.4 第 3 家 · 寒武纪 MLU590（已完成）

| 项 | 状态 | 结果 |
|---|---|---|
| 阶段 0–1 环境 + 接入 | ✅ | `cambricon` backend 落地；conformance **13/13 + 6/6**；smoke **42/0** |
| 阶段 2 训练腿 | ✅ | 2 卡 DDP（`cncl`）+ 三类通信对照：**6/6** · loss **15.4498 → 11.1479** · **2957.8 tok/s**（⚠️ CNCL 走 MLU_LINK 片间互联，**非 RDMA**，已如实标注） |
| 阶段 3 多流 16 项基线 | ✅ | **15 通过 / 1 不适用 / 0 不支持**；探针 **8/8**（含 S-13）、图捕获 **5/5**、S-16 配额 **3/3**；与另两家**仅 1 项差异**（S-12 流优先级，**与 P800 相反**） |
| 阶段 4 错误闭环 | ✅ | 四类注入 **5 闭环 / 0 跳过 / 0 失败** |
| 能力声明（如实） | ✅ | `error_map` **不声明** —— 实测 CNRT 抛**错误名而非数字码**，无法建码表（原"预期可做出比 P800 更完整码表"已被实测否定） |
| 阶段 5 收敛 | 🔄 | 接入方案 + 16 项基线已产出；推理腿（前向 + 服务化）未做 |

### 3.5 原型构建进度（可量化）

| 维度 | 现状 |
|---|---|
| 抽象契约 | **13 个 `@abstractmethod`**（设备 4 · 多流 7 · 错误 1 · 恢复 1）+ 可选方法 4 个 |
| 已接入后端 | 3 个（`ascend` / `kunlun` / `cambricon`），**均为厂商官方 torch 插件路线**（`torch_npu` / XPytorch / `torch_mlu`） |
| 原型代码量 | runtime 侧 **4618 行**（芯片无关） |
| 能力矩阵 | 声明能力数 **13 / 8 / 9**；**13 项能力中普遍可满足仅 7 项**（device · memory · stream · event · multidevice · recovery_probe · device_state），其余 6 项必须显式降级或如实声明 |
| 判据集 | conformance **13 例 + 推理 6 例**，三后端共用同一套 |
| 跨后端缺陷 | **10 例**（审计台账第 6–15 条）"只有非首实例才暴露"的框架缺陷，全部修在**框架层** |
| 标准化资产 | 统一启动脚本 `serve_standard.sh`（两实例真机均 `SERVE_STANDARD_PASS`）· 环境普查脚本 `preflight_env.sh` |
| 文档 | 规范/手册/标准 **13 份** + 芯片专属文档 · 复核清单（9 条命令） |
| 证据 | 910C **8 份** / P800 **65 份**原始证据，含 09-22 对称复跑 `recheck_*` |

### 3.5 下一步计划

```text
① 第 3 家寒武纪 MLU590 —— ✅ **已完成（2026-09-22）**
   环境打通 → 镜像定档 → backends/cambricon/ 落地 → conformance 13/13 + 6/6
        → 多流 16 项（15 通过 / 1 不适用）→ 训练腿 6/6（2957.8 tok/s，`dist=cncl`）
        → 错误闭环 5/0/0 → 产出并入接入手册 SOP + 接口修订建议
   已顺带验证：接口约定修订建议第 1 条（device_type 与 vendor 分离）
              —— `mlu` 是第三种命名空间，多命名空间场景首次真实检验 ✔

② 并行推进（不依赖第 3 家）
   接口缺口的修复闭环（3 条待裁定 → 落地）+ 契约不变式判据 I1–I4 实现
   镜像入锁诉求推进（第 2 家建议以官方 `-base` 入锁）
   接入手册随第 3 家实践迭代

③ 2026.10 及以后
   P800 第二实例已在 9 月完成 → 10 月转设备上下文核心机制 + 20 模型设备侧支撑
   11 月 多芯片 conformance 收敛 + 页锁定内存 + 双缓冲流水线
   12 月 多芯片稳定性看护 + 拓扑感知传输 + 组件 v0.3
```

---

## 附：外部口径

1. **三家芯片的推进顺序**：第 1 家 昇腾 910C（已完成）→ 第 2 家 昆仑芯 P800（已完成）→ 第 3 家 寒武纪（进行中）。
2. **验收判据**：统一的 conformance 判据集 + 多流 16 项基线 + 两条腿证据链；
   **判据本身不随芯片变化**——这正是"统一设备抽象封装"成立的证明。
