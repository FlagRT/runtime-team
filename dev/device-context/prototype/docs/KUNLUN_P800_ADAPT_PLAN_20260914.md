# 昆仑芯 P800 适配工作方案（2026-09-14）

> 目标：把 910C 上已验证的**设备上下文 + 多流 Stream** 能力适配到昆仑芯 P800，
> **基于既有统一原型接入**（`prototype/runtime/backends/` 插件机制），不另起炉灶。
> **当前状态**：阶段 0（环境与单卡基线）**已完成**；阶段 1（`kunlun` backend 实现）**待方案确认后启动**。
>
> 配套文档：
> - 环境汇总 → [`KUNLUN_P800_ENV_REPORT_20260914.md`](KUNLUN_P800_ENV_REPORT_20260914.md)
> - **五域基线实测（缺失项清单在此）** → [`KUNLUN_P800_BASELINE_PROBE_20260914.md`](KUNLUN_P800_BASELINE_PROBE_20260914.md)
> - 探针脚本与原始结果 → [`../../probes/kunlun/`](../../probes/kunlun/)

---

## 0. 一页速览

| 项 | 状态 |
|---|---|
| 昆仑芯资源 | ✅ 可用：8× P800（96 GB/卡）、1.5 TiB 内存、384 线程；⚠️ **共享机**（25 人在线、22 容器） |
| 容器 | ✅ `hliu553-device-context-p800` 运行中（镜像 `flaggems-main-dev:202608`，本机镜像库已有，无需 pull） |
| 调用入口 | `conda activate python310_torch29_cuda` + `export XPU_EVENT_KL3_ENABLE=1`；**设备 API 走 `torch.cuda`**（见 §2.1） |
| 单卡五域基线 | 设备抽象 ✅ ｜ 多流 Stream/Event ✅ ｜ 算子 ✅ ｜ 错误 ⚠️ ｜ 状态恢复 ⏳ |
| 已识别缺失项 | **3 条**：1 项上游缺陷 + 1 项上游约束 + 1 项设计依据（见基线报告 §3） |
| 距"适配完成"还差 | ① `kunlun` backend 实现 ② conformance ③ 训练腿（多卡）④ 推理腿 ⑤ 错误闭环 ⑥ 对外提交 |
| **下一步** | **等你确认本方案 → 启动阶段 1（开始写 `backends/kunlun/`）** |

---

## 1. 我方职责边界（先把"我们负责什么"说清）

### 1.1 我方负责（对应 910C 已交付的能力）

| 域 | 内容 | 昆仑芯需实现的对应项 |
|---|---|---|
| **设备抽象** | `device_count` / `set_device` / `memory_stats` / `probe_device` | XPU 设备枚举、显存查询、探活（实测走 `torch.cuda`） |
| **多流 Stream** | `create_stream` / `create_event` / `current_stream` / `stream_context` / `synchronize` / `synchronize_stream` / `wait_event_host`，及 16 项流语义子项 | XPU Stream/Event 封装；**有界同步**待测、跨流可见性已实测正确 |
| **错误码翻译** | `translate_error`：错误 → L1–L4 分级 + `disposition` | **厂商错误码在 Python 层不可得** → 映射表以「异常类型 + 消息模板」为键（见基线报告 §3.2） |
| **状态恢复** | `recover_device(probe/real/hybrid)` | 设备级重置原语**待测** |
| **插件机制（我们起头）** | `RuntimeBackend` 抽象基类 + 注册表 + 统一 conformance | 新增 `kunlun` backend；**新增芯片 = 实现 backend + 跑通 conformance** |

**接入契约**：实现 13 个 `@abstractmethod`（上表五域）+ 可选 `stream_priority_range()`；
经 `supports()` 如实声明能力边界（不支持项不伪造，conformance 会如实跳过）。

> **已对着代码核实（2026-09-14）**：`prototype/runtime/backends/base.py` 实测**恰好 13 个 `@abstractmethod`**，
> 与上表五域一一对应 —— 设备抽象 4（`device_count` / `set_device` / `memory_stats` / `probe_device`）、
> 多流 Stream 7（`create_stream` / `create_event` / `current_stream` / `synchronize` /
> `stream_context` / `synchronize_stream` / `wait_event_host`）、
> 错误码翻译 1（`translate_error`）、状态恢复 1（`recover_device`）。
> 且 `backends/registry.py` 的 `_KNOWN_BACKENDS` **已预留 `("ascend", "flagos", "kunlun")`** ——
> 昆仑芯的接入位在原型设计时已留好，本次属**填位**而非改造。
> conformance 侧现状：`cases.py`（13 例）、`infer_cases.py`（6 例），
> 已有结果集 `conformance_runtime_ascend.json`（13）、`conformance_runtime_infer.json`（6）、`conf_proto_flagos_13.json`（13）。

### 1.2 不属我方（识别到即**对外提交**，不自行适配）

| 缺失项类型 | 归属子方向 | 我方提供什么 |
|---|---|---|
| 算子缺失/精度问题（FlagGems、flagtree 算子） | 算子适配方向 | 复现脚本 + 失败用例 + 参数 |
| 集合通信（xccl / flagcx 在 XPU 上的实现） | 分布式方向 | 通信对照失败记录（哪类操作、什么错误） |
| 显存池 / 缓存管理 | 显存与缓存管理方向 | 显存峰值与碎片观测数据 |
| 任务调度 / 执行引擎 | 任务调度方向 | 调度行为异常记录 |
| 监控诊断 / 多级恢复编排 | 监控诊断方向 | 设备侧分级与恢复原语结果 |
| 精度与性能分析（Oracle-Device / Profiling） | 精度与性能方向 | 统一口径的性能与精度原始数据 |
| 镜像 / 基座（驱动、SDK 版本、镜像发布） | 总组 / 镜像负责方 | 环境信息汇总 + 复现条件 |

### 1.3 归属判定规则（跑训练时逐条套用）

1. 失败点落在 `RuntimeBackend` 五域内 → **我方适配**；
2. 在五域之外 → **对外提交**，提交单须含：现象、最小复现、错误原文、初步定位证据；
3. 无法判定 → 先记入缺失项清单，在下次周会（或每周三 STATUS.md）提请定位。

---

## 2. 接入方式（复用既有原型，不另起炉灶）

```
prototype/runtime/backends/
├── ascend/     # 910C 推理腿（torch_npu）——已验证 13/13 + 6/6
├── flagos/     # 910C 训练腿（torch_fl）——已验证 13/13
└── kunlun/     # 【本次新增】昆仑芯 XPU
```

- **代码复用**：`errors.py`（统一错误对象与分级）、`recovery.py`（恢复五段式）、
  `device_state.py`（四态）、`api/stream.py`（Stream/Event 统一封装）**直接复用**，
  昆仑芯只需实现设备与流事件的原生映射。
- **conformance 复用**：`runtime/conformance/runner.py --backend kunlun` 直接跑同一套
  13 例 + 推理 6 例，作为接入完备度的度量器（910C 上 FlagOS 后端从零到 13/13 即先例）。
- **不修改**既有后端与统一 API（插件式，互不影响）。

### 2.1 接入路线：内部用厂商命名空间，统一放在我方抽象层

**调研依据**（2026-09-14 核对 xliu969 在 P800 上的既有工作与全组路线决策）

| 项 | 事实 |
|---|---|
| 该方向在 P800 上的活跃子方向 | **`dev/memory`（显存与缓存管理）**；其 `dev/device-context/` 内是 910C 的 FlagCX/HCCL 补丁，`compose.base.yml` 是昇腾底座（CANN 9.0 + `/dev/davinci*`） |
| P800 设备栈 | **厂商 CUDA 兼容 torch（xpytorch）** + `torch_xray 2.0.4` + `xmlir` + `xtorch_ops`；XCCL/BKCL 为 FlagCX klx 底层 |
| **torch 编译标志（原文）** | `USE_CUDA=ON, USE_CUDNN=ON, USE_NCCL=1, USE_XCCL=OFF, **USE_XPU=OFF**` |
| **选卡变量** | **`CUDA_VISIBLE_DEVICES`**（实测 `=2` → 1 卡；`=2,5` → 2 卡） |
| 其运行环境变量口径 | `VLLM_PLUGINS=fl`、`VLLM_FL_PLATFORM=kunlunxin`、`USE_FLAGGEMS=1`、`GEMS_VENDOR=kunlunxin`、`KLX_USE_AUTOTUNE=0` |
| **全组路线决策** | **2026-08-22：生产交付统一走路线 A（各芯片厂商设备插件）+ FlagGems + FlagCX + vllm-plugin-FL；2026-09-03 `torch_fl` 设备层路线冻结** |
| 决策原话 | 「"跨芯统一设备层"目前不是 B 的现实优势，而是其最薄弱处」 |

**结论：不自造 `torch.xpu` 包装。三点理由**

1. **与全组已定路线冲突**：自建统一设备层正是被冻结的路线 B 的思路；路线 A 的语义就是「直接用厂商插件」，
   在厂商层之上再造一个设备层，等于把刚冻结的方向重建一遍。
2. **技术上不是「包装」而是「重建」**：`USE_XPU=OFF` ⇒ `torch.xpu` 无编译实现（`is_available()` 实测 False）。
   要让 `torch.xpu` 可用，需自行实现设备枚举 / 流 / 事件 / 内存分配 / 错误码 / 通信对接全套 —— 这是自建设备层。
3. **会造成两套设备世界割裂**：FlagGems（实测 dispatch key = **CUDA**）、FlagCX/BKCL、vllm-plugin-FL
   全挂在 **CUDA 设备语义**上；我方的 Stream/Event 若走独立命名空间，就无法与它们在同一上下文里正确串流与同步。

**因此：统一点放在我方抽象层，而非 torch 命名空间层**

```
统一 API（我方）   runtime/api/{stream,errors}.py + RuntimeBackend 抽象 + conformance
      ↑ 各后端内部各用各的厂商命名空间（互不影响）
├── ascend   → torch_npu
├── flagos   → torch_fl
└── kunlun   → torch.cuda（xpytorch）   ← 仅内部实现选择，对外统一 API 与 conformance 不变
```

**收益**：若厂商日后提供可用的 `torch.xpu`（`USE_XPU=ON`），**只改 `kunlun` backend 内部实现**，
对外 API 与 conformance 用例**一行不动** —— 这正是 backend 插件化的价值所在。

**附：一处既有判断偏差，建议与分布式/显存方向对齐时提示更正**
xliu969 的 P800 实测文档记有「torch.xpu 亦存在 → **双通道**」。该结论来自 `hasattr(torch,'xpu')`
（其探针 `p800_env_check.py` 仅打印该属性是否存在），**未测 `is_available()`**。
我方实测 `torch.xpu.is_available() = False`，且编译标志为 `USE_XPU=OFF`
⇒ **功能上并非双通道，实际只有 CUDA 单通道**。

### 2.2 后端划分依据：**厂商实现**，而非 torch 命名空间

`ascend` / `flagos` / `kunlun` 这三个名字是**我们注册表里的键**（`_KNOWN_BACKENDS`），
与 torch 命名空间**不是一一对应关系**：

| 我们的后端名（对外） | 对应厂商 | **内部 torch 命名空间** | 厂商插件 / 运行时 |
|---|---|---|---|
| `ascend` | 昇腾 | `torch.npu` | torch_npu / CANN |
| `flagos` | 昇腾（走设备层路线 B） | `torch_fl` 自有 API | torch_fl（**已冻结**） |
| `kunlun` | 昆仑芯 | **`torch.cuda`** | XPytorch / torch_xray / XCCL |
| （未来）`nvidia` | NVIDIA | `torch.cuda` | 官方 PyTorch / NCCL |

**⚠️ 由此产生的设计约束：`kunlun` 与未来的 `nvidia` 共用 `torch.cuda` 命名空间。**
因此 **`supports()` / `probe_vendor()` 不能靠命名空间判别厂商**，必须用厂商特征：

| 判别特征 | 昆仑芯 P800 | NVIDIA |
|---|---|---|
| 厂商 Python 模块 | `torch_xray` / `torch_xmlir` 存在 | 无 |
| 宿主伪文件系统 | **`/proc/kunlun/` 存在** | 无 |
| 宿主工具 | `xpu-smi` 存在 | `nvidia-smi` 存在 |
| `get_device_name(0)` | `"GPU"` | 真实型号（如 `NVIDIA A100-SXM4-80GB`） |
| 通信库 | `libbkcl.so`（XCCL/BKCL） | `libnccl.so` |

**并且两者的 `torch.cuda` 行为并不等价**（昆仑芯是兼容 / 重定向层）：
流优先级在此栈触发 PyTorch `INTERNAL ASSERT`、厂商错误码不透出
（见基线报告 §3.1 / §3.2）。⇒ **不能把 `kunlun` 当作 `nvidia` 的别名**，能力集必须分开声明。

**另一处命名债（低优先，建议月度评估时一并处理，现在不建议动）**
现有三个键混了两个轴：`ascend` / `kunlun` 是**芯片**，`flagos` 是**设备层路线**。
若要统一为「按芯片命名」，改动会波及已有 conformance 结果文件与文档指针。

---

## 3. 验证要求与验收标准

### 3.1 验证矩阵（910C 基线 vs P800 目标）

| 层级 | 项目 | 910C 基线 | P800 目标 / 判据 |
|---|---|---|---|
| 组件自检 | `smoke_runtime.py` | 37/37 | 通过率不低于基线；差异项**逐条说明**（不支持项如实跳过） |
| 接口完备度 | conformance 13 例 + 推理 6 例 | ascend 13/13 + 6/6；flagos 13/13 | **全绿或如实跳过**；每个跳过项计入缺失项清单 |
| 多流语义 | 16 项流语义子项 | 逐项 ✅ / 如实跳过 | 同上；**流优先级已知不支持**（上游缺陷，基线报告 §3.1） |
| 训练腿 | 最小分布式训练（2 卡起） | loss 15.4497→11.15、2117 tok/s、通信三类对照全对 | 通信对照三类（all_reduce / all_gather / P2P）各 1 组通过、loss 单调下降、无 NaN、吞吐已记录 |
| 推理腿 | 单卡推理（前向 + 服务化） | 区分度 0.4123、108 句/s、p50 27.4 ms | 维度正确、无 NaN、区分度可分辨、吞吐与 p50 已记录 |
| 错误闭环 | 注入 → 分级 → 恢复 | 推理腿 5 闭环 / 训练腿 4 闭环 | 闭环项数 / 总项；**厂商错误码不可得项须如实标注** |

> 训练腿与推理腿**权重对等**，并行推进，不存在主辅关系。

### 3.2 已实测的环境约束（对标 910C 踩过的坑）

| 项 | 910C 基线 | P800 实测 | 影响 |
|---|---|---|---|
| 驱动 / SDK | CANN 9.0 | 宿主 `xpu-smi` **5.0.21.47**；容器内 **515.58**（正常分层） | 版本已锁定，可写入约束清单 |
| **设备 API 入口** | `torch_npu` / `torch_fl` | **`torch.cuda`**（`USE_XPU=OFF`，`torch.xpu.is_available()=False`） | **后端实现必须绑 `torch.cuda`** |
| **设备可见性** | `ASCEND_RT_VISIBLE_DEVICES` | **`CUDA_VISIBLE_DEVICES`**（实测 `=2` → 1 卡；`=2,5` → 2 卡） | 多卡隔离用法已确定 |
| 卡间互联 | HCCL / RoCE | **XPU0-3（NUMA0）、XPU4-7（NUMA1）组内 XL 私有链路；跨组 SYS** | 训练腿规模设计：**优先组内配对**可避开跨 NUMA |
| 网卡与卡亲和 | — | **NIC0-3 ↔ XPU0-3、NIC4-7 ↔ XPU4-7 均为 PIX**，200 Gb RoCE，`PORT_ACTIVE` | 多卡通信路径干净 |
| 网卡直访显存 | 910C 待查（HIXL 是否等价 GDR） | **`kunlun_peermem` 已加载** | 昆仑芯侧存在类 GDR 原语 → 纳入跨芯片原语调研 |
| 显存 / 内存 / CPU | 64 GB / — / — | **96 GB × 8 / 1.5 TiB（无 swap）/ 384 线程** | 充裕 |
| 集合通信后端 | HCCL | `flagcx`、`xccl`、`kccl` 均注册；`libbkcl.so` 随进程加载 | 训练腿可复用 |
| **镜像落盘** | 数据盘 11 T | `/var/lib/docker` **已 bind mount 到 `/data1`（5.8 T NVMe，剩 1.5 T）** | ✅ 充足，**不是**阻塞项 |
| 机器共享程度 | 独占 | **25 人在线、22 容器、272 僵尸进程** | ⚠️ 共享机，见 §5.1 |

> 早前误判已更正：曾判「docker data-root 在只剩 2.9 GB 的根分区、镜像必然加载失败」，
> 经 `findmnt -T /var/lib/docker` 查实为误（误因：`du -x` 遇跨文件系统即停止）。详见环境报告 §5.3。

### 3.3 验收标准：什么算"P800 适配完成"

**以下 6 条须同时满足**

| # | 判据 |
|---|---|
| 1 | `backends/kunlun/` 实现全部 **13 个抽象方法**；`supports()` 声明与实测一致，**不伪造能力** |
| 2 | conformance **13 例 + 推理 6 例**全绿**或如实跳过**；每个跳过项在报告中说明原因与归属 |
| 3 | `smoke_runtime.py` 通过率不低于 910C 基线，或差异项已逐条解释 |
| 4 | **训练腿**：最小分布式训练跑通，通信三类对照通过，loss 单调下降、无 NaN、吞吐已记录 |
| 5 | **推理腿**：单卡推理跑通，维度 / 无 NaN / 区分度 / 吞吐 / p50 指标齐备 |
| 6 | **错误闭环**跑通；缺失项清单完成**归属判定**；非我方项已提交或已明确提交路径 |

**明确不在本次范围**（避免范围蔓延）：
- **多机多卡**（本机仅单机 8 卡，无第二台 P800）；
- 精度与性能的深度调优（属精度与性能方向）；
- `torch.xpu` 可用性改造（属上游，且与全组路线冲突，见 §2.1）。

---

## 4. 执行计划

### 4.1 阶段与里程碑

| 阶段 | 时间 | 交付 | 对应验收 |
|---|---|---|---|
| **阶段 0 · 环境与基线** | 2026-09-14 ✅ **已完成** | 环境汇总、五域基线实测、接入路线调研 | — |
| **阶段 1 · 单卡接入** | 9/14–9/18（本周） | `backends/kunlun/` + `supports()` + smoke + conformance 13/6 | 验收 1–3 |
| **阶段 2 · 训练腿（多卡）** | 9/21–9/25 | 最小分布式训练（2 卡起，优先 XPU0-1 组内配对）+ 通信三类对照 | 验收 4 |
| **阶段 3 · 推理腿** | 9/21–9/25（**与阶段 2 并行**） | 单卡前向 + 服务化，复用 `Qwen3-Embedding-0.6B` | 验收 5 |
| **阶段 4 · 错误闭环** | 9/28–9/30 | 昆仑芯错误映射表 v0（异常类型 + 消息模板为键）+ 注入→分级→恢复闭环 | 验收 6 |
| **阶段 5 · 收敛** | 9/30 前 | 适配记录、缺失项清单（含归属）、对外提交单、STATUS.md 更新 | 全部 6 条 |

### 4.2 进度表（截至 2026-09-14 11:00）

| # | 步骤 | 状态 | 说明 |
|---|---|---|---|
| 1 | 环境信息汇总 | ✅ **已完成** | [`KUNLUN_P800_ENV_REPORT_20260914.md`](KUNLUN_P800_ENV_REPORT_20260914.md) |
| 2 | 起容器 + 装组件 | ✅ **已完成** | 容器 `hliu553-device-context-p800` 运行中 |
| 2a | └ 镜像获取 | ✅ **无需拉取** | 本机镜像库已有 `flaggems-main-dev:202608`（38.3 GB）→ 手册的 59.9 GB pull / 32 GB load 全部跳过 |
| 2b | └ 容器启动 | ✅ **已完成** | 对齐本机已跑通配置（非 privileged / bridge / `--shm-size=64g` / `/dev/xpu0..7`+`xpuctrl`+`fuse`），挂 `/data2/hliu553:/workspace` |
| 2c | └ flagtree | ✅ **xpu3.6 已满足** | 镜像内含 `flagtree 0.6.1+xpu3.6`；手册最新 `0.7.0rc1+xpu3.6` → 升级列为可选实验（§9） |
| 2d | └ FlagGems | ✅ **已满足** | 镜像内含 `flag_gems 5.3.4`，算子级实测通过（`add` max diff = 0.0）；源码在容器内 `/env/FlagGems` |
| 2e | └ 五域基线实测 | ✅ **已完成** | [`KUNLUN_P800_BASELINE_PROBE_20260914.md`](KUNLUN_P800_BASELINE_PROBE_20260914.md) |
| 3 | **阶段 1**：`kunlun` backend + conformance | ⏳ **待方案确认** | **尚未动代码**（遵循「方案确认后再实现」） |
| 4 | **阶段 2/3**：训练腿 + 推理腿 | ⏳ 待执行（前置已解除） | 可立即启动；训练腿是**识别缺失项**的主手段 |
| 5 | **阶段 4/5**：错误闭环 + 收敛 | ⏳ 待执行 | 缺失项清单已出 3 条；对外提交单 2 张待起草（§6） |

---

## 5. 风控

### 5.1 已知风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| **共享机竞争**（25 人在线、22 容器） | 训练吞吐抖动、显存被抢 | 跑前 `xpu-smi` 确认空闲；固定卡用 `CUDA_VISIBLE_DEVICES`；记录跑测时段便于复现 |
| 上游**流优先级缺陷** | 该子项无法验证 | 如实跳过 + 对外提交（§6.1）；**不等它阻塞整体进度** |
| 上游**厂商错误码不透出** | 错误分级置信度下降 | 映射表以「异常类型 + 消息模板」为键；相关条目标 `is_grade_confident=False` |
| **同进程错误污染** | 错误闭环实验相互干扰 | **每个失败用例独立进程**（已固化进探针脚本） |
| `github.com` 容器内**不可达** | 无法 clone 依赖 | 用容器内既有 `/env/FlagGems`（HEAD `73c5aff1`）；必要时改用镜像或本地 tar |
| 宿主 **`dma_excp_mask = 0`** | 复杂算子可能触发 KL3 kernel 异常（status 719） | 记为待验证项（§10）；若复现再评估是否需 sudo 置 1（须与平台沟通） |
| flagtree 版本与手册不一致 | 与官方推荐口径有差异 | 保持镜像组合为基线；升级作为**单变量实验**（§9） |
| 仅单机 8 卡，无多机 | 验证范围受限 | §3.3 已明确把多机排除在范围外 |

### 5.2 已踩过的坑与教训（备查）

| 坑 | 教训 |
|---|---|
| `ssh P800` 连不上（`Connection refused`） | 真实端口是 **26008**；`~/.ssh/config` 缺 `Port` 行。**连不上的第一件事是核对端口与 `ssh -G <host>` 实际生效配置**，别先怀疑网络策略 |
| 误判「镜像无处落盘」 | **判断某目录占多少盘，先 `findmnt -T` / `df -T` 定位文件系统**，不要用 `du -x` 的差值反推（`-x` 遇跨文件系统即停止） |
| 误读「设备级错误粘滞」 | 隔离成独立进程后不复现 ⇒ 污染只在**同进程内**；涉及错误串扰的实验必须进程隔离 |
| 容器内 `python3` 不是目标环境 | 默认是 conda **base 3.13.11**；必须 `conda activate python310_torch29_cuda`（3.10.18） |
| 容器内 `github.com` 超时 | 不要默认联网可用；先测通路再决定依赖获取方式 |

---

## 6. 对外提交物（非我方职责项）

| # | 提交项 | 归属 | 最小复现 | 状态 |
|---|---|---|---|---|
| **6.1** | `Stream.priority_range()` 触发 PyTorch `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188` | XPytorch / 昆仑芯 torch 后端的 stream priority 实现 | `python3 -c "import torch; print(torch.cuda.Stream.priority_range())"`（需先激活环境 + `XPU_EVENT_KL3_ENABLE=1`） | ⏳ 待起草 |
| **6.2** | 厂商错误码不透出到 Python 异常（`AcceleratorError` 消息无码；仅退出钩子偶见 `error code= 101`） | 上游错误上报层 | `python3 -c "import torch; torch.cuda.set_device(99)"` 观察异常消息 | ⏳ 待起草 |

**提交单须含**：现象、最小复现、错误原文、初步定位证据、影响面（我方哪个能力被卡）。
**提交渠道待确认**（见 §10 第 4 项）。

---

## 7. 交付物清单

| 交付物 | 位置 | 状态 |
|---|---|---|
| 环境汇总 | `prototype/docs/KUNLUN_P800_ENV_REPORT_20260914.md` | ✅ 已交付 |
| 五域基线实测报告 | `prototype/docs/KUNLUN_P800_BASELINE_PROBE_20260914.md` | ✅ 已交付 |
| 探针脚本与原始结果 | `dev/device-context/probes/kunlun/` | ✅ 已交付 |
| 本方案 | `prototype/docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` | ✅ 已交付 |
| **昆仑芯后端实现** | `prototype/runtime/backends/kunlun/` | ⏳ 阶段 1 |
| conformance 结果 | `prototype/runtime/conformance/kunlun_*.json` | ⏳ 阶段 1 |
| 训练腿 / 推理腿结果 | `prototype/runtime/proto/kunlun_*.json` | ⏳ 阶段 2/3 |
| 适配记录（含缺失项与归属） | `prototype/docs/KUNLUN_ADAPT_RECORD_<date>.md` | ⏳ 阶段 5 |
| 对外提交单 | 按 §6 渠道 | ⏳ 阶段 5 |
| 本方向状态更新 | `dev/device-context/STATUS.md`（每周三） | 🔄 持续 |

---

## 8. 可复用资产与版本差异

| 资产 | 位置 | 说明 |
|---|---|---|
| flagtree xpu3.6 镜像包（32 GiB） | `/data1/dinghaisong/flagtree-xpu3.6-...-**202606**-base.tar.gz` | 全局可读；但**本机镜像库已加载其对应镜像**，无需再用 |
| **本机镜像库（推荐）** | `flaggems-main-dev:202608`（38.3 GB）、`ubuntu22.04:202606-base`（34.7 GB） | 直接可用，**零下载成本** |
| FlagGems 源码 | 容器内 `/env/FlagGems`（HEAD `73c5aff1`） | 来自官方 clone，**无需联网** |
| **共享模型缓存** | `/data1/dinghaisong/hf_cache`（1.7 T，可读） | 含 **`Qwen3-Embedding-0.6B`（1.2 G，实测可读）** —— 正是原型验收模型 |
| runtime-team 代码（他人） | `/data2/xliu969/code/runtime-team` | 可读参考；其 P800 活跃子方向为 `dev/memory` |

---

## 9. 可选实验（非阻塞）：flagtree 0.6.1 → 0.7.0rc1

**现状**：镜像内含 **`flagtree 0.6.1+xpu3.6`**（xpu3.6 后端已满足），且已实测**算子级可用**。
官方手册的免源码安装行给出的是 **`flagtree===0.7.0rc1+xpu3.6`**。

**为什么不立即升级**：镜像的 `flagtree + flag_gems + torch` 是一套**被验证过的组合**，
贸然升级会把「接入验证」与「版本升级」两个变量混在一起，违反单变量原则。

```bash
# 与基线隔离：先快照现状，再升级，复跑同一探针对比
python3 -m pip freeze > /workspace/pin_before_$(date +%Y%m%d_%H%M).txt
python3 -m pip uninstall -y triton          # 反复执行至卸净
python3 -m pip install flagtree===0.7.0rc1+xpu3.6 \
  --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple
# 复跑同一探针，逐项对比 /workspace/dc_probe_isolated_result.json
```

**判据**：① 8 卡仍可见；② FlagGems 算子 max diff 仍为 0；③ **流优先级缺陷是否消失**（§6.1）；
④ 无新增报错。任一不满足即回滚（容器可重建，回滚成本为零）。

**来源**：官方手册 [User manual for xpu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu)。
网络实测：`resource.flagos.net` 与 `pypi.tuna.tsinghua.edu.cn` 容器内**可达（HTTP 200）**，
`github.com` **不可达**（超时）。

---

## 10. 待补测与待确认清单

| # | 问题 | 状态 | 实测命令 / 判据 |
|---|---|---|---|
| 1 | **设备级重置 / 重建**原语是否存在 | ⏳ 待测 | 探 `torch.cuda` 是否有 device reset / context 重建接口；影响 `recover_device` 的 real 模式 |
| 2 | **有界等待**能力（`synchronize(timeout)`） | ⏳ 待测 | 910C 有（pyACL `synchronize_stream_with_timeout`）、FlagOS 无；昆仑芯待确认，决定 `supports()` 如何声明 |
| 3 | **带卡容器并发上限** | ⏳ 待测 | 910C 为 3（超限 `acl.init()` 返 500000）；若存在需提总组入约束清单 |
| 4 | **厂商错误码**是否有其他可达通道 | 🔶 部分回答 | Python 异常层不可得（§6.2）；待确认是否有 C++ 层日志/环境变量可开 |
| 5 | 对外提交的**渠道** | ⏳ 待确认 | 昆仑芯支持渠道？还是经总组转达？影响 §6 两项的落地 |
| 6 | 跨流 Event 依赖 | ✅ **已回答** | 实测 `cross_stream_visible = 45056.0 == 期望值`，语义正确 |
| 7 | 设备可见性变量 | ✅ **已回答** | 即 `CUDA_VISIBLE_DEVICES`（`=2` → 1 卡；`=2,5` → 2 卡） |
| 8 | 单机多卡规模与卡间通信 | ✅ **已回答** | 单机 8× P800（96 GB/卡）；组内 XL、跨组 SYS；8× 200 G RoCE，NIC 与卡 PIX 直连，`kunlun_peermem` 已加载 |
| 9 | 容器内 `xpu-smi` 版本 | ✅ **已回答** | **515.58**（宿主 5.0.21.47，属正常分层） |
