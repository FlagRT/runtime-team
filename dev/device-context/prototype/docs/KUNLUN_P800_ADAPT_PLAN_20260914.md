# 昆仑芯 P800 接入工作方案（2026-09-14）

> **定位**：以昆仑芯 P800 作为统一运行时原型的**第二个接入实例**，
> 按《运行时层接口约定 · 设备上下文章节》（`INTERFACE_CONTRACT_DC_20260908.md`）§2 的
> **Backend 插件接入规范**新建 `kunlun` backend，实证该规范在**非昇腾芯片**上成立；
> 再用分布式训练 / 推理两条腿暴露问题、**先做简单修复**，完成后把原型 release 给运行时层其他子方向。
>
> **⚠️ 本方案不是「把 910C 的能力适配到 P800」。** 910C 提供的是**规范与方法**；
> 其**实现、错误码表、环境约束均不迁移**，P800 按自己的厂商栈（`torch.cuda` / xpytorch）新建。
>
> 配套文档：
> - 环境汇总 → [`KUNLUN_P800_ENV_REPORT_20260914.md`](KUNLUN_P800_ENV_REPORT_20260914.md)
> - **五域基线实测（缺失项清单在此）** → [`KUNLUN_P800_BASELINE_PROBE_20260914.md`](KUNLUN_P800_BASELINE_PROBE_20260914.md)
> - 探针脚本与原始结果 → [`../../probes/kunlun/`](../../probes/kunlun/)

---

## 0. 一页速览

| 项 | 状态 |
|---|---|
| **本方案在做什么** | 按**接入规范**为原型新建第二个芯片实例（昆仑芯），并暴露训推问题后简单修复 |
| 昆仑芯资源 | ✅ 8× P800（96 GB/卡）、1.5 TiB 内存、384 线程；⚠️ **共享机**（25 人在线、22 容器） |
| 容器 | ✅ `hliu553-device-context-p800` 运行中（镜像 `flaggems-main-dev:202608`，本机镜像库已有，无需 pull） |
| 调用入口 | `conda activate python310_torch29_cuda` + `export XPU_EVENT_KL3_ENABLE=1`；**设备 API 走 `torch.cuda`** |
| **阶段 1 接入结果** | ✅ **已完成**：`kunlun` backend 落地；conformance **13/13** + 推理 **6/6**；`smoke_runtime.py` **42 通过 / 0 失败** |
| 单卡五域基线 | 设备抽象 ✅ ｜ 多流 Stream/Event ✅ ｜ 算子 ✅ ｜ 错误 ⚠️（无厂商码）｜ 状态恢复 ⚠️（仅 probe 级） |
| 已识别问题 | **上游 3 条**（基线报告 §3）**＋ 接入过程暴露并已修 3 条**（§7.5） |
| 距"本方向职责完成"还差 | ① 训推验证与简单修复 ②《新芯片接入手册》③ 规范修订建议 ④ release |
| **下一步** | 阶段 2/3：训练腿（多卡）+ 推理腿并行 |

---

## 1. 定位与交付边界

### 1.1 两阶段口径（重要，避免与月度计划冲突）

| 阶段 | 时间 | 范围 | 谁负责 |
|---|---|---|---|
| **原型接入（当前）** | **2026.09** | 统一运行时**原型搭建** + **接入的头和框架**；**昇腾与昆仑芯两个实例**都在本轮 | **本方向（device-context）** |
| 运行时层最终交付 | 2026.11 | 整个运行时层的交付（含 ≥2 家新芯片接入） | 运行时层各子方向共同 |

> 依据：月度计划 2026.11 行「≥2 家新芯片接入（昆仑芯 + 寒武纪，单个 ≤5 人天）；接入手册」，
> 与接口约定 §2「接入成本目标 ≤5 人天（11 月昆仑芯/寒武纪实做验证）」。
> **本方案把昆仑芯这一实例的接入工作放在 9 月完成**，使 11 月的整体交付有可复用的框架与手册。

### 1.2 我们交付什么、release 给谁

**我们负责的是"头和框架"**，不是把每个芯片做深。交付链条：

```
按接入规范新建 kunlun backend  →  跑 conformance 验证接口完备度
        ↓
用分布式训练 / 推理两条腿暴露问题  →  属五域的"先简单修复"；不属五域的对外提交
        ↓
产出《新芯片接入手册》+ 规范修订建议
        ↓
★ release 给运行时层其他子方向（验证 + 迭代循环优化）
```

**"先简单修复"的边界**（避免范围蔓延）：
- ✅ 修：`RuntimeBackend` 五域内的问题（接口映射、能力声明、错误分级、恢复路径、流语义）
- ❌ 不修：算子缺失/精度（→ 算子方向）、集合通信实现（→ 分布式方向）、显存池（→ 显存方向）、
  性能深度调优（→ 精度与性能方向）；这些一律**对外提交**
- ❌ 不做：把原型打磨到生产级（那是 release 之后各子方向的迭代循环）

### 1.3 规范 vs 实现：什么迁移、什么不迁移

| 类别 | 内容 | 在 P800 上怎么处理 |
|---|---|---|
| **规范（章程）** | 统一 API 与语义承诺、五域抽象、Backend 插件接入规范、两条硬纪律（`record_stream` / 错误隔离分层）、错误 L1–L4 分级与 `disposition` 处置约定、`recover_device` 返回契约、错误分级调用纪律、conformance 判据（13+6）、验收口径（smoke / conformance） | ✅ **必须遵守**，逐条落地 |
| **方法** | 最小变更 / 单变量隔离、证据规范（命令 + 数值）、错误闭环验证法、六项归属判定规则 | ✅ **复用方法**，不复用其结论 |
| **规范的载体（框架代码）** | `runtime/api/`、`backends/base.py` 的 13 个抽象方法、`conformance/` 13+6 例 | ✅ 这是规范的可执行形式，本就为多后端设计 |
| **910C 的落地实例** | `backends/ascend/` 与 `backends/flagos/` 的实现与绑定、**108 条 ACL 错误码映射表**、CANN 相关约束、镜像基座、并发上限 3、`ASCEND_RT_VISIBLE_DEVICES` | ❌ **不迁移**；P800 按 `torch.cuda` / xpytorch **新建** |

**一句话**：迁移的是**规范与方法**，不迁移的是**实现与结论**。

---

## 2. 我方职责边界

### 2.1 我方负责（规范五域在昆仑芯上的落地）

| 域 | 内容 | 昆仑芯需实现的对应项（现状） |
|---|---|---|
| **设备抽象** | `device_count` / `set_device` / `memory_stats` / `probe_device` | 走 `torch.cuda`（实测 8 卡、96 GiB、`mem_get_info` 准确） |
| **多流 Stream** | `create_stream` / `create_event` / `current_stream` / `stream_context` / `synchronize` / `synchronize_stream` / `wait_event_host`，及 16 项流语义子项 | 跨流 Event 依赖**已实测正确**；**有界同步待测**；**流优先级不支持**（上游缺陷） |
| **错误码翻译** | `translate_error`：错误 → L1–L4 + `disposition` | **厂商错误码在 Python 层不可得** → 映射表以「异常类型 + 消息模板」为键（基线报告 §3.2） |
| **状态恢复** | `recover_device(probe/real/hybrid)` | 设备级重置原语**待测** |
| **插件机制（我们起头）** | `RuntimeBackend` 抽象 + 注册表 + 统一 conformance | 新增 `kunlun` backend；**新增芯片 = 实现 backend + 跑通 conformance** |

**接入契约**：实现 13 个 `@abstractmethod` + 可选 `stream_priority_range()`；
经 `supports()` 如实声明能力边界（不支持项不伪造，conformance 会如实跳过）。

> **已对着代码核实（2026-09-14）**：`backends/base.py` 实测**恰好 13 个 `@abstractmethod`**：
> 设备抽象 4（`device_count` / `set_device` / `memory_stats` / `probe_device`）、
> 多流 Stream 7（`create_stream` / `create_event` / `current_stream` / `synchronize` /
> `stream_context` / `synchronize_stream` / `wait_event_host`）、
> 错误码翻译 1（`translate_error`）、状态恢复 1（`recover_device`）。
> `registry.py` 的 `_KNOWN_BACKENDS` **已预留 `("ascend", "flagos", "kunlun")`** ——
> 接入位在原型设计时已留好，本次属**填位**而非改造。
> conformance 现状：`cases.py`（13 例）、`infer_cases.py`（6 例），已有 result json 三份。

### 2.2 不属我方（识别到即**对外提交**，不自行适配）

| 缺失项类型 | 归属子方向 | 我方提供什么 |
|---|---|---|
| 算子缺失/精度问题（FlagGems、flagtree 算子） | 算子适配方向 | 复现脚本 + 失败用例 + 参数 |
| 集合通信（xccl / flagcx 在 XPU 上的实现） | 分布式方向 | 通信对照失败记录（哪类操作、什么错误） |
| 显存池 / 缓存管理 | 显存与缓存管理方向 | 显存峰值与碎片观测数据 |
| 任务调度 / 执行引擎 | 任务调度方向 | 调度行为异常记录 |
| 监控诊断 / 多级恢复编排 | 监控诊断方向 | 设备侧分级与恢复原语结果 |
| 精度与性能分析（Oracle-Device / Profiling） | 精度与性能方向 | 统一口径的性能与精度原始数据 |
| 镜像 / 基座（驱动、SDK 版本、镜像发布） | 总组 / 镜像负责方 | 环境信息汇总 + 复现条件 |

### 2.3 归属判定规则（逐条套用）

1. 失败点落在 `RuntimeBackend` 五域内 → **我方先简单修复**；
2. 在五域之外 → **对外提交**，提交单须含：现象、最小复现、错误原文、初步定位证据；
3. 无法判定 → 先记入缺失项清单，在下次周会（或每周三 STATUS.md）提请定位。

---

## 3. 接入方式：按规范新建 backend

```
prototype/runtime/backends/
├── ascend/     # 910C 推理腿（torch_npu）——已验证 13/13 + 6/6
├── flagos/     # 910C 训练腿（torch_fl）——已验证 13/13
└── kunlun/     # 【本次新建】昆仑芯 P800 —— 第二个芯片实例
```

**接入步骤（照接口约定 §2 逐条做）**

| # | 步骤 | 说明 |
|---|---|---|
| 1 | 新建 `runtime/backends/kunlun/`，实现 `RuntimeBackend` 五域抽象 | 内部绑 `torch.cuda`；复用 `errors.py` / `recovery.py` / `device_state.py` / `api/stream.py` 的**框架** |
| 2 | 提供 `build()` 工厂函数，在 `registry` 登记名字 `kunlun` | `_KNOWN_BACKENDS` 已预留 |
| 3 | `supports()` **如实声明**能力边界 | 已知：流优先级不支持；有界同步待测后声明 |
| 4 | 跑通 conformance：`python3 runtime/conformance/runner.py --backend kunlun` | 13 例 + 6 例**全绿或 stub-skip 报告说明缺口**即接入完成 |
| 5 | 跑 `smoke_runtime.py` 自检 | 组件质量基线对齐 910C 的 37/37 |

**不修改**既有后端与统一 API（插件式，互不影响）。

### 3.1 接入路线：内部用厂商命名空间，统一放在我方抽象层

**调研依据**（2026-09-14 核对 xliu969 在 P800 上的既有工作与全组路线决策）

| 项 | 事实 |
|---|---|
| 该方向在 P800 上的活跃子方向 | **`dev/memory`（显存与缓存管理）**；其 `dev/device-context/` 内是 910C 的 FlagCX/HCCL 补丁，`compose.base.yml` 是昇腾底座 |
| P800 设备栈 | **厂商 CUDA 兼容 torch（xpytorch）** + `torch_xray 2.0.4` + `xmlir` + `xtorch_ops`；XCCL/BKCL 为 FlagCX klx 底层 |
| **torch 编译标志（原文）** | `USE_CUDA=ON, USE_CUDNN=ON, USE_NCCL=1, USE_XCCL=OFF, **USE_XPU=OFF**` |
| **选卡变量** | **`CUDA_VISIBLE_DEVICES`**（实测 `=2` → 1 卡；`=2,5` → 2 卡） |
| 其运行环境变量口径 | `VLLM_PLUGINS=fl`、`VLLM_FL_PLATFORM=kunlunxin`、`USE_FLAGGEMS=1`、`GEMS_VENDOR=kunlunxin`、`KLX_USE_AUTOTUNE=0` |
| **全组路线决策** | **2026-08-22：生产交付统一走路线 A（各芯片厂商设备插件）+ FlagGems + FlagCX + vllm-plugin-FL；2026-09-03 `torch_fl` 设备层路线冻结** |

**结论：不自造 `torch.xpu` 包装。三点理由**

1. **与全组已定路线冲突**：自建统一设备层正是被冻结的路线 B 的思路；路线 A 的语义就是「直接用厂商插件」。
2. **技术上不是「包装」而是「重建」**：`USE_XPU=OFF` ⇒ `torch.xpu` 无编译实现（`is_available()` 实测 False）。
   要让它可用，需自行实现设备枚举 / 流 / 事件 / 内存分配 / 错误码 / 通信对接全套。
3. **会造成两套设备世界割裂**：FlagGems（实测 dispatch key = **CUDA**）、FlagCX/BKCL、vllm-plugin-FL
   全挂在 **CUDA 设备语义**上；我方的 Stream/Event 若走独立命名空间，无法与它们在同一上下文里正确串流同步。

**统一点放在我方抽象层，而非 torch 命名空间层**

```
统一 API（我方）   runtime/api/{stream,errors}.py + RuntimeBackend 抽象 + conformance
      ↑ 各后端内部各用各的厂商命名空间（互不影响）
├── ascend   → torch_npu
├── flagos   → torch_fl
└── kunlun   → torch.cuda（xpytorch）
```

> **附：一处既有判断偏差，建议对齐时提示更正**
> xliu969 的 P800 实测文档记有「torch.xpu 亦存在 → 双通道」，该结论仅来自 `hasattr(torch,'xpu')`
> （其探针未测 `is_available()`）。我方实测 + 编译标志 `USE_XPU=OFF` 表明 **只有 CUDA 单通道**。

### 3.2 后端划分依据：**厂商实现**，而非 torch 命名空间

`ascend` / `flagos` / `kunlun` 是我们注册表里的**键**，与 torch 命名空间**不是一一对应**：

| 后端名（对外） | 厂商 | **内部 torch 命名空间** | 厂商插件 / 运行时 |
|---|---|---|---|
| `ascend` | 昇腾 | `torch.npu` | torch_npu / CANN |
| `flagos` | 昇腾（设备层路线 B） | `torch_fl` 自有 API | torch_fl（**已冻结**） |
| `kunlun` | 昆仑芯 | **`torch.cuda`** | XPytorch / torch_xray / XCCL |
| （未来）`nvidia` | NVIDIA | `torch.cuda` | 官方 PyTorch / NCCL |

**⚠️ 设计约束：`kunlun` 与未来的 `nvidia` 共用 `torch.cuda`。**
因此 **`supports()` / `probe_vendor()` 不能靠命名空间判别厂商**，必须用厂商特征：

| 判别特征 | 昆仑芯 P800 | NVIDIA |
|---|---|---|
| 厂商 Python 模块 | `torch_xray` / `torch_xmlir` 存在 | 无 |
| 宿主伪文件系统 | **`/proc/kunlun/` 存在** | 无 |
| 宿主工具 | `xpu-smi` 存在 | `nvidia-smi` 存在 |
| `get_device_name(0)` | `"GPU"` | 真实型号（如 `NVIDIA A100-SXM4-80GB`） |
| 通信库 | `libbkcl.so`（XCCL/BKCL） | `libnccl.so` |

**且两者 `torch.cuda` 行为不等价**（昆仑芯是兼容 / 重定向层）：
流优先级触发 PyTorch `INTERNAL ASSERT`、厂商错误码不透出。⇒ **不能把 `kunlun` 当 `nvidia` 的别名**。

**命名债（低优先，建议月度评估时处理）**：现有三个键混了两个轴 ——
`ascend` / `kunlun` 是**芯片**，`flagos` 是**设备层路线**。统一命名会波及已有 conformance 结果文件与文档指针。

---

## 4. 验证要求与验收标准

### 4.1 验证矩阵（910C 基线 vs P800 目标）

| 层级 | 项目 | 910C 基线 | P800 目标 / 判据 |
|---|---|---|---|
| 组件自检 | `smoke_runtime.py` | 37/37 | 通过率不低于基线；差异项**逐条说明** |
| 接口完备度 | conformance 13 例 + 推理 6 例 | ascend 13/13 + 6/6；flagos 13/13 | **全绿或如实跳过**；每个跳过项计入缺失项清单并标归属 |
| 多流语义 | 16 项流语义子项 | 逐项 ✅ / 如实跳过 | 同上；**流优先级已知不支持**（上游缺陷） |
| 训练腿 | 最小分布式训练（2 卡起） | loss 15.4497→11.15、2117 tok/s、通信三类对照全对 | 通信三类（all_reduce / all_gather / P2P）各 1 组通过、loss 单调下降、无 NaN、吞吐已记录 |
| 推理腿 | 单卡推理（前向 + 服务化） | 区分度 0.4123、108 句/s、p50 27.4 ms | 维度正确、无 NaN、区分度可分辨、吞吐与 p50 已记录 |
| 错误闭环 | 注入 → 分级 → 恢复 | 推理腿 5 闭环 / 训练腿 4 闭环 | 闭环项数 / 总项；**厂商错误码不可得项须如实标注** |

> 训练腿与推理腿**权重对等**，并行推进；两者都是**暴露问题**的手段，不是性能验收。

### 4.2 已实测的环境约束

| 项 | 910C 基线 | P800 实测 | 影响 |
|---|---|---|---|
| 驱动 / SDK | CANN 9.0 | 宿主 `xpu-smi` **5.0.21.47**；容器内 **515.58** | 版本已锁定 |
| **设备 API 入口** | `torch_npu` / `torch_fl` | **`torch.cuda`**（`USE_XPU=OFF`） | 后端实现必须绑 `torch.cuda` |
| **设备可见性** | `ASCEND_RT_VISIBLE_DEVICES` | **`CUDA_VISIBLE_DEVICES`**（实测 `=2` → 1 卡） | 多卡隔离用法已确定 |
| 卡间互联 | HCCL / RoCE | **XPU0-3（NUMA0）、XPU4-7（NUMA1）组内 XL；跨组 SYS** | 训练腿**优先组内配对**避开跨 NUMA |
| 网卡与卡亲和 | — | **NIC0-3 ↔ XPU0-3、NIC4-7 ↔ XPU4-7 均 PIX**，200 Gb RoCE | 多卡通信路径干净 |
| 网卡直访显存 | 待查（HIXL 是否等价 GDR） | **`kunlun_peermem` 已加载** | 纳入跨芯片原语调研 |
| 显存 / 内存 / CPU | 64 GB / — / — | **96 GB × 8 / 1.5 TiB（无 swap）/ 384 线程** | 充裕 |
| 集合通信后端 | HCCL | `flagcx`、`xccl`、`kccl` 均注册；`libbkcl.so` 随进程加载 | 训练腿可复用 |
| **镜像落盘** | 数据盘 11 T | `/var/lib/docker` **已 bind mount 到 `/data1`（剩 1.5 T）** | ✅ 充足，**不是**阻塞项 |
| 机器共享程度 | 独占 | **25 人在线、22 容器、272 僵尸进程** | ⚠️ 共享机，见 §6.1 |

> 早前误判已更正：曾判「docker data-root 在只剩 2.9 GB 的根分区、镜像必然加载失败」，
> 经 `findmnt -T /var/lib/docker` 查实为误（误因：`du -x` 遇跨文件系统即停止）。详见环境报告 §5.3。

### 4.3 验收标准：什么算"本方向职责完成"

**以下 6 条须同时满足**

| # | 判据 |
|---|---|
| 1 | `backends/kunlun/` 实现全部 **13 个抽象方法**；`supports()` 声明与实测一致，**不伪造能力** |
| 2 | conformance **13 例 + 推理 6 例**全绿**或如实跳过**；每个跳过项说明原因与归属 |
| 3 | `smoke_runtime.py` 通过率不低于 910C 基线，或差异项已逐条解释 |
| 4 | **训练腿**：最小分布式训练跑通，通信三类对照通过，loss 单调下降、无 NaN；暴露的问题已按 §2.3 分流（五域内已修 / 五域外已提交） |
| 5 | **推理腿**：单卡推理跑通，维度 / 无 NaN / 区分度 / 吞吐 / p50 指标齐备 |
| 6 | **产出齐备**：《新芯片接入手册》+ 接口约定修订建议 + 缺失项清单（含归属）；**原型可 release 给其他子方向** |

**明确不在本次范围**（避免范围蔓延）：
- **多机多卡**（本机仅单机 8 卡，无第二台 P800）；
- 精度与性能的深度调优（属精度与性能方向）；
- 把原型打磨到生产级（属 release 之后各子方向的迭代循环）；
- `torch.xpu` 可用性改造（属上游，且与全组路线冲突）。

---

## 5. 执行计划

### 5.1 阶段与里程碑

| 阶段 | 时间 | 交付 | 对应验收 |
|---|---|---|---|
| **阶段 0 · 环境与基线** | 2026-09-14 ✅ **已完成** | 环境汇总、五域基线实测、接入路线调研 | — |
| **阶段 1 · 接入** | 9/14 ✅ **已完成** | `backends/kunlun/` + `supports()` + `build()` 登记 + smoke + conformance **13/13 + 6/6** | 验收 1–3 ✅ |
| **阶段 2 · 训练腿** | 9/21–9/25 | 最小分布式训练（2 卡起，优先 XPU0-1 组内配对）+ 通信三类对照 + 问题分流 | 验收 4 |
| **阶段 3 · 推理腿** | 9/21–9/25（**与阶段 2 并行**） | 单卡前向 + 服务化，复用 `Qwen3-Embedding-0.6B` | 验收 5 |
| **阶段 4 · 错误闭环** | 9/21–9/30（穿插） | 昆仑芯错误映射表 v0（异常类型 + 消息模板为键）+ 注入→分级→恢复闭环 | 验收 4–5 |
| **阶段 5 · 产出与 release** | 9/28–9/30 | 《新芯片接入手册》+ 规范修订建议 + 缺失项清单 + 对外提交单 + STATUS 更新 + **release** | 验收 6 |

### 5.2 进度表（截至 2026-09-14 11:00）

| # | 步骤 | 状态 | 说明 |
|---|---|---|---|
| 1 | 环境信息汇总 | ✅ **已完成** | [`KUNLUN_P800_ENV_REPORT_20260914.md`](KUNLUN_P800_ENV_REPORT_20260914.md) |
| 2 | 起容器 + 装组件 | ✅ **已完成** | 容器 `hliu553-device-context-p800` 运行中 |
| 2a | └ 镜像获取 | ✅ **无需拉取** | 本机镜像库已有 `flaggems-main-dev:202608`（38.3 GB）→ 59.9 GB pull / 32 GB load 全部跳过 |
| 2b | └ 容器启动 | ✅ **已完成** | 对齐本机已跑通配置（非 privileged / bridge / `--shm-size=64g` / `/dev/xpu0..7`+`xpuctrl`+`fuse`），挂 `/data2/hliu553:/workspace` |
| 2c | └ flagtree | ✅ **xpu3.6 已满足** | 镜像内含 `flagtree 0.6.1+xpu3.6`；升级列为可选实验（§9） |
| 2d | └ FlagGems | ✅ **已满足** | 镜像内含 `flag_gems 5.3.4`，算子级实测通过（`add` max diff = 0.0）；源码在 `/env/FlagGems` |
| 2e | └ 五域基线实测 | ✅ **已完成** | [`KUNLUN_P800_BASELINE_PROBE_20260914.md`](KUNLUN_P800_BASELINE_PROBE_20260914.md) |
| 3 | **阶段 1**：`kunlun` backend + conformance | ✅ **已完成** | `backends/kunlun/` 已实现；conformance **13/13 + 6/6**；smoke **42/0**；证据见 §7.1 |
| 3a | └ 接入过程修复 | ✅ **已修 3 项** | registry 急切求值缺陷、conformance f1 厂商码假设、smoke 仅覆盖昇腾（详见 §7.5） |
| 4 | **阶段 2/3**：训练腿 + 推理腿 | ⏳ 待执行（前置已解除） | 可立即启动；训练腿是**暴露问题**的主手段 |
| 5 | **阶段 4/5**：错误闭环 + 产出与 release | ⏳ 待执行 | 缺失项清单已出 3 条；对外提交单 2 张待起草（§7.4） |

---

## 6. 风控

### 6.1 已知风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| **共享机竞争**（25 人在线、22 容器） | 训练吞吐抖动、显存被抢 | 跑前 `xpu-smi` 确认空闲；固定卡用 `CUDA_VISIBLE_DEVICES`；记录跑测时段便于复现 |
| 上游**流优先级缺陷** | 该子项无法验证 | 如实跳过 + 对外提交（§7.4）；**不等它阻塞整体进度** |
| 上游**厂商错误码不透出** | 错误分级置信度下降 | 映射表以「异常类型 + 消息模板」为键；相关条目标 `is_grade_confident=False` |
| **同进程错误污染** | 错误闭环实验相互干扰 | **每个失败用例独立进程**（已固化进探针脚本） |
| `github.com` 容器内**不可达** | 无法 clone 依赖 | 用容器内既有 `/env/FlagGems`（HEAD `73c5aff1`）；必要时改用镜像或本地 tar |
| 宿主 **`dma_excp_mask = 0`** | 复杂算子可能触发 KL3 kernel 异常（status 719） | 记为待验证项（§10）；若复现再评估是否需 sudo 置 1（须与平台沟通） |
| flagtree 版本与手册不一致 | 与官方推荐口径有差异 | 保持镜像组合为基线；升级作为**单变量实验**（§9） |
| 仅单机 8 卡，无多机 | 验证范围受限 | §4.3 已明确把多机排除在范围外 |

### 6.2 已踩过的坑与教训（备查）

| 坑 | 教训 |
|---|---|
| `ssh P800` 连不上（`Connection refused`） | 真实端口是 **26008**；config 缺 `Port` 行。**连不上的第一件事是核对端口与 `ssh -G <host>` 实际生效配置** |
| 误判「镜像无处落盘」 | **判断某目录占多少盘，先 `findmnt -T` / `df -T` 定位文件系统**，不要用 `du -x` 差值反推 |
| 误读「设备级错误粘滞」 | 隔离成独立进程后不复现 ⇒ 污染只在**同进程内**；错误串扰实验必须进程隔离 |
| 容器内 `python3` 不是目标环境 | 默认是 conda **base 3.13.11**；必须 `conda activate python310_torch29_cuda`（3.10.18） |
| 容器内 `github.com` 超时 | 不要默认联网可用；先测通路再决定依赖获取方式 |
| 把本方案目标写成"把 910C 能力适配到 P800" | **910C 只提供规范与方法，实现与结论不迁移**；P800 是按规范新建的**第二个实例** |

---

## 7. 产出物

### 7.1 交付物清单

| 交付物 | 位置 | 状态 |
|---|---|---|
| 环境汇总 | `prototype/docs/KUNLUN_P800_ENV_REPORT_20260914.md` | ✅ 已交付 |
| 五域基线实测报告 | `prototype/docs/KUNLUN_P800_BASELINE_PROBE_20260914.md` | ✅ 已交付 |
| 探针脚本与原始结果 | `dev/device-context/probes/kunlun/` | ✅ 已交付 |
| 本方案 | `prototype/docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` | ✅ 已交付 |
| **昆仑芯后端实现** | `prototype/runtime/backends/kunlun/`（`backend.py` + `__init__.py`） | ✅ 阶段 1 |
| **conformance 结果** | `prototype/runtime/conformance/conformance_runtime_kunlun.json`（13/13）、`..._kunlun_infer.json`（6/6） | ✅ 阶段 1 |
| **组件自检证据** | `dev/device-context/probes/kunlun/smoke_kunlun_20260914.txt`（42 通过 / 0 失败） | ✅ 阶段 1 |
| 训练腿 / 推理腿结果 | `prototype/runtime/proto/kunlun_*.json` | ⏳ 阶段 2/3 |
| **《新芯片接入手册》** | `prototype/docs/NEW_CHIP_ONBOARD_GUIDE.md` | ⏳ 阶段 5 |
| **接口约定修订建议** | `prototype/docs/INTERFACE_CONTRACT_REVISION_<date>.md` | ⏳ 阶段 5 |
| 适配记录（含缺失项与归属） | `prototype/docs/KUNLUN_ADAPT_RECORD_<date>.md` | ⏳ 阶段 5 |
| 本方向状态更新 | `dev/device-context/STATUS.md`（每周三） | 🔄 持续 |

### 7.2 《新芯片接入手册》提纲（月度计划 2026.11 的交付物）

| 章 | 内容 | 素材来源 |
|---|---|---|
| 1 | 接入总览：新增一家芯片 = 实现 backend + 跑通 conformance | 接口约定 §2 |
| 2 | 五域抽象逐项怎么填（含可选能力如何声明） | 接口约定 §1 + 本次实操 |
| 3 | **厂商判别与命名空间映射**（同命名空间不同厂商如何区分） | 本方案 §3.2 |
| 4 | 环境准备清单（容器、设备节点、环境变量、选卡变量） | 环境报告 + 本方案 §4.2 |
| 5 | conformance 跑法与判读（全绿 / stub-skip 怎么写缺口说明） | 本次实操 |
| 6 | **常见坑**（本方案 §6.2 + 昆仑芯特有：`torch.xpu` 空壳、默认 python 是 base、github 不可达） | 本次实操 |
| 7 | 归属判定：什么问题自己修、什么问题对外提交 | 本方案 §2.3 |
| 8 | 验收清单（6 条判据） | 本方案 §4.3 |

### 7.3 接口约定修订建议（本次待收集，阶段 5 汇总）

P800 是接口约定**首次被非昇腾芯片检验**，已识别出需要斟酌的规范点：

| 规范点 | 观察到的问题 | 建议方向 |
|---|---|---|
| `supports()` 能力声明的**粒度** | 昆仑芯「流优先级」有接口但不可用（上游缺陷） | 规范需区分「不支持」与「有接口但有缺陷」两种声明语义 |
| `synchronize_stream(timeout_ms)` **有界同步** | 规范写「超时抛 TimeoutError」，但未定义**底层不支持有界同步时**的降级语义 | 补充降级条款（如降级为无界 + 显式标注能力缺失） |
| `translate_error` 的 `graded_by` | 规范含 `code_map` / `message_hint`；昆仑芯**无错误码可得** | 明确「仅 message_hint 可用」时的置信度标注要求（`is_grade_confident`） |
| **厂商判别**要求 | 规范未提「两家厂商可能共用同一 torch 命名空间」 | 在接入规范中补「厂商判别」条款（§3.2 特征表） |
| **conformance 用例的设备无关性** | ① `f1` 原断言硬要求厂商错误码非空，超出其自称的「类别/位置/根因三投影」；② `t3` 提示文案写死 `torch_npu` | 用例只应依赖统一 API 与 `supports()`；提示文案避免写死厂商名（`f1` 已修，见 §7.5） |
| **注册流程的副作用** | `registry` 注册日志会触发后端 `info()`，而 `info()` 常需加载厂商依赖；缺依赖时**中断整个发现流程** | 规范要求后端 `info()` 无副作用，且注册/发现不得依赖 `info()` 成功（已修，见 §7.5） |
| **判据的后端覆盖** | 原 `smoke_runtime.py` 只显式覆盖昇腾，其他后端不被自检 | 判据应设备无关：任何已注册后端都能被同一套自检覆盖（已修，见 §7.5） |

### 7.4 对外提交物（非我方职责项）

| # | 提交项 | 归属 | 最小复现 | 状态 |
|---|---|---|---|---|
| **7.4.1** | `Stream.priority_range()` 触发 PyTorch `INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188` | XPytorch / 昆仑芯 torch 后端的 stream priority 实现 | `python3 -c "import torch; print(torch.cuda.Stream.priority_range())"`（需先激活环境 + `XPU_EVENT_KL3_ENABLE=1`） | ⏳ 待起草 |
| **7.4.2** | 厂商错误码不透出到 Python 异常（`AcceleratorError` 消息无码；仅退出钩子偶见 `error code= 101`） | 上游错误上报层 | `python3 -c "import torch; torch.cuda.set_device(99)"` | ⏳ 待起草 |

**提交单须含**：现象、最小复现、错误原文、初步定位证据、影响面（我方哪个能力被卡）。**提交渠道待确认**（§10）。

### 7.5 本次接入暴露并已修的问题（供《新芯片接入手册》收录）

阶段 1 落地过程中，昆仑芯这一「非昇腾实例」暴露了 3 个**与具体芯片无关的框架/判据缺陷**，
均在本方向五域内，已按「先简单修复」处理：

| # | 问题 | 现象与根因 | 修复 | 回归证据 |
|---|---|---|---|---|
| **1** | **registry 注册日志急切求值** | `logger.debug("...: %s", backend.info())` 的 `info()` **无条件被调用**；`flagos.info()` → `_load()` → `import torch_fl`，在 P800 上抛 `ModuleNotFoundError`，**穿透 `register()` 中断整个 `discover()`**，违背该模块自称的「发现失败仅告警、不中断」 | ① 日志改为 `if logger.isEnabledFor(logging.DEBUG)` 守卫 + `try/except`；② 把 `factory()` 构造与 `register()` 一并纳入 `discover()` 容错 | 修复后 `discover()` 在 P800 上返回 `['ascend','flagos','kunlun']` 且不中断；smoke 42/0 |
| **2** | **conformance `f1` 硬要求厂商错误码** | 原断言 `fe.error_code is not None` 超出用例自称的「类别/位置/根因三投影」契约，把昇腾/flagos 的 `ret=XXXX` 当成通用前提 → **无厂商码的后端恒 FAIL，与实现质量无关** | 按 `supports("error_map")` 分支：声明者仍要求 `error_code` 非空；未声明者改为要求 `mapped=False` 且类别/根因正确 | **向后兼容已实测**：同一无码错误下，模拟「声明 error_map」的后端仍判 FAIL（既有断言路径一字未改），昆仑芯判 PASS |
| **3** | **`smoke_runtime.py` 只覆盖昇腾** | 第 [5] 节硬编码 `ascend`，非昇腾后端得不到自检覆盖 | 新增第 **[6] 节「真实后端通用自检（后端无关）」**：按注册表自动挑选可用后端，跑厂商无关契约检查；新增 `--backend` 参数（不传则自动挑选，保持原用法可用）；能力相关项按 `supports()` 如实判定 | `--backend kunlun` 与无参数两种调用均为 **42 通过 / 0 失败** |

> **这三条正是「P800 作为规范首个非昇腾实例」的价值所在**：它们与昆仑芯本身无关，
> 但在昇腾单实例下永远暴露不出来。已一并写入 §7.3 修订建议，供接口约定升版参考。

---

## 8. 可复用资产与版本差异

| 资产 | 位置 | 说明 |
|---|---|---|
| **本机镜像库（推荐）** | `flaggems-main-dev:202608`（38.3 GB）、`ubuntu22.04:202606-base`（34.7 GB） | 直接可用，**零下载成本** |
| flagtree xpu3.6 镜像包 | `/data1/dinghaisong/flagtree-xpu3.6-...-**202606**-base.tar.gz` | 全局可读；本机镜像库已加载其对应镜像，无需再用 |
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

**判据**：① 8 卡仍可见；② FlagGems 算子 max diff 仍为 0；③ **流优先级缺陷是否消失**（§7.4.1）；
④ 无新增报错。任一不满足即回滚（容器可重建，回滚成本为零）。

**来源**：官方手册 [User manual for xpu](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-xpu)。
网络实测：`resource.flagos.net` 与 `pypi.tuna.tsinghua.edu.cn` 容器内**可达（HTTP 200）**，`github.com` **不可达**。

---

## 10. 待补测与待确认清单

| # | 问题 | 状态 | 实测命令 / 判据 |
|---|---|---|---|
| 1 | **设备级重置 / 重建**原语是否存在 | ⏳ 待测 | 探 `torch.cuda` 是否有 device reset / context 重建接口；影响 `recover_device` 的 real 模式 |
| 2 | **有界等待**能力（`synchronize(timeout)`） | ⏳ 待测 | 910C 有（pyACL）、FlagOS 无；昆仑芯待确认，决定 `supports()` 如何声明（并回填 §7.3 修订建议） |
| 3 | **带卡容器并发上限** | ⏳ 待测 | 910C 为 3（超限 `acl.init()` 返 500000）；若存在需提总组入约束清单 |
| 4 | **厂商错误码**是否有其他可达通道 | 🔶 部分回答 | Python 异常层不可得（§7.4.2）；待确认是否有 C++ 层日志/环境变量可开 |
| 5 | 对外提交的**渠道** | ⏳ 待确认 | 昆仑芯支持渠道？还是经总组转达？影响 §7.4 两项的落地 |
| 6 | 跨流 Event 依赖 | ✅ **已回答** | 实测 `cross_stream_visible = 45056.0 == 期望值`，语义正确 |
| 7 | 设备可见性变量 | ✅ **已回答** | 即 `CUDA_VISIBLE_DEVICES`（`=2` → 1 卡；`=2,5` → 2 卡） |
| 8 | 单机多卡规模与卡间通信 | ✅ **已回答** | 单机 8× P800（96 GB/卡）；组内 XL、跨组 SYS；8× 200 G RoCE，NIC 与卡 PIX 直连，`kunlun_peermem` 已加载 |
| 9 | 容器内 `xpu-smi` 版本 | ✅ **已回答** | **515.58**（宿主 5.0.21.47，属正常分层） |
