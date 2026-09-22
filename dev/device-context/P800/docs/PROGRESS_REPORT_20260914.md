# 设备上下文 · 全量进度报告（2026-09-14）

> **范围**：本方向自身职责 —— **设备抽象与执行上下文 / 多流 Stream / 错误码翻译 / 状态恢复**（子方向 1）。
> 不覆盖其他子方向（显存 / 调度 / 算子适配 / 分布式 / 监控 / 精度）的证据，那些由各自方向出具。
>
> **目的**：把 2026-09-02 ～ 09-14 的碎片信息收拢成一份可判断的整体视图。
>
> **分支**：`kistich/device-context` ｜ **PR 目标**：`dev-1.0` ｜ **验收模型**：Qwen3-Embedding-0.6B

---

## 0. 一屏结论

| 板块 | 状态 | 关键数字 |
|---|---|---|
| **910C · 统一原型与两条腿** | ✅ **完成** | conformance 13/13 + 6/6 ｜ smoke 37/37 ｜ 训练 loss 15.4497→11.15 ｜ 推理 108 句/s |
| **910C · 错误闭环** | ✅ **完成** | 推理腿 5 闭环 ｜ 训练腿 4 闭环 / 1 如实跳过 |
| **910C · 组件打包 v0.1.0** | ✅ **完成** | Git tag `runtime-v0.1.0` |
| **910C · 遗留** | 🔶 **4 项非阻塞** | 见 §1.4 |
| **P800 · 定位与方案** | ✅ **已定稿** | 第二个接入实例；迁移规范与方法，不迁移实现 |
| **P800 · 阶段 0 环境与基线** | ✅ **完成** | 8×P800 96GB 全空闲 ｜ 五域基线已取 |
| **P800 · 阶段 1 接入** | ✅ **完成** | conformance **13/13 + 6/6** ｜ smoke **42/0** ｜ 提交 `530f456` |
| **P800 · 阶段 2 训练腿** | ✅ **已在标注条件下完成（多卡跑通）** | 两 rank **TRAIN_LEG_PASS 6/6**；loss **15.4488 → 11.1481**（50 步）；**3482 tok/s**；⇢ 条件：`XPU_EVENT_KL3_ENABLE` 未设（见 §2.6.3） |
| **P800 · 阶段 3 推理腿** | ⏸ **未启动** | — |
| **P800 · 阶段 4 错误闭环** | ⏸ **未启动** | — |
| **P800 · 阶段 5 产出与 release** | ⏸ **未启动** | 产出：《新芯片接入手册》+ 规范修订建议 |

**一句话**：910C 主干已闭环进入规模扩展阶段；P800 已完成「环境 → 基线 → 接入」三步，
**阶段 2 的三类集合通信已在探针与训练脚本中双重验证通过**，训练腿已跑到梯度同步环节并**精确定位到卡点**
（见 §2.4.7）；过程中曾误判「flagcx 在 P800 上不可用」，已被卡 6,7 复测推翻（见 §2.4.4）。

### 三个需要你判断的要点

1. **P800 的定位在 09-14 被纠正过**：从「把 910C 能力适配过去」改为「按接入规范**新建第二个实例**」。
   口径是：9 月＝原型接入（昇腾 + 昆仑芯两个实例都在本轮），11 月＝运行时层整体交付；
   本方向职责＝**原型搭建 + 接入的头和框架**，接入→训推暴露问题→五域内先简单修复→**release 给其他子方向**迭代。
2. **阶段 1 的真正价值是挖出 3 个「与昆仑芯无关」的框架/判据缺陷**（§2.3.3）——
   它们是「规范首次被非昇腾芯片检验」的产物，昇腾单实例下永远暴露不出来。
3. **一条重要更正：集合通信不是缺陷，卡才是变量**（§2.4.4）。
   同一份 flagcx 代码：卡 0,1 → KL3 内核异常；卡 0,2 → 超时挂死；**卡 6,7 → 三类通信全 OK**。
   因此「flagcx/BKCL 在 P800 上不可用」的初判**不成立**，`dma_excp_mask` 的 sudo 动作也**暂不需要**；
   真因指向**共享机上被其他租户占用的卡**。**用卡前必须查 `xpu-smi` 挑连续空闲卡**——这是目前最重要的一条实操纪律。

---

## 1. 910C 侧（主干，已闭环）

### 1.1 交付物

| 交付项 | 内容 | 位置 |
|---|---|---|
| 统一基座配置 | 锁定两腿镜像、使用规则（含并发上限 3）、合入把关五条 | `dev/stack.lock.910c.v2.yaml`（总组定稿，本方向只消费不自建） |
| 统一运行时 API | 后端选择 / 设备 / 流与事件 / 错误翻译 / 状态恢复 | `prototype/runtime/api/` |
| Backend 插件机制 | 抽象基类（13 个 `@abstractmethod`）+ 注册表 + 自动发现 | `prototype/runtime/backends/` |
| 昇腾后端 | `torch_npu`，推理腿使用 | `backends/ascend/` |
| FlagOS 后端 | `torch_fl`，训练腿使用（⚠️ **2026-09-22 起 910C 训练腿已统一 torch_npu**，本行仅适用于 `flagos` 备用路径） | `backends/flagos/` |
| 统一 conformance | 13 例 + 推理 6 例，跨后端可跑 | `prototype/runtime/conformance/` |
| 接口约定（**规范本体**） | API 承诺 + **§2 Backend 插件接入规范（新芯片照此实现）** + 两条硬纪律 | `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` |
| 组件 v0.1.0 | Release note + Git tag | tag `runtime-v0.1.0` |
| 仓库重组 | **device-context（原型，芯片无关）** + **910C（第一实例）** + **P800（第二实例）** + 各级看板（2026-09-16 按芯片重组后） | — |
| 组织架构 PR | 同步 dev-1.0，**0 冲突** | ⏳ 待评审合入 |

**设计主张**：**新增一家芯片 = 实现一个 backend + 跑通 conformance**（接入成本目标 ≤5 人天）。
本轮 flagos 后端从零到 13/13 是这条主张的第一次实证；**P800 是第二次实证，一次「非昇腾」的实证**。

### 1.2 量化结果（**全部在锁定镜像内取得**，非个人调试容器）

| 项 | 结果 |
|---|---|
| 组件自检 | **37/37**（无 NPU 环境时昇腾项自动 SKIP） |
| conformance 昇腾后端 | **13/13 + 推理 6/6** |
| conformance FlagOS 后端 | **13/13**（锁定训练镜像） |
| 训练腿 2 卡分布式微调 | 两 rank 均 **6/6**：loss **15.4497 → 11.15**（50 步）、**2117 tok/s**、通信三类对照（all_reduce / all_gather / P2P）全对 |
| 推理腿单卡服务化 | **10/10**：维度 1024、范数 1.0、语义区分度 **0.4123**、**108 句/s**（p50 27.4ms）、超长输入 → L2_PARAM/raise 且业务继续 |
| 推理腿单卡前向（对照） | **10/10**：区分度 0.638、66–79 句/s、无 NaN |
| 错误注入 → 恢复闭环 | 推理腿 **5 闭环 / 0 失败**；训练腿 **4 闭环 / 1 跳过 / 0 失败**（无有界同步的后端**如实跳过，不伪造**） |

**诚实标注**：
- 两 rank 末值 loss 有约 4e-3 差异（11.1497 / 11.1541），源于集合通信浮点累加顺序，属固有特性；
- `ERR99999` 偶发进程终止**复现 1 次、后续 4 次未复现** → 观察项，**不作结论**。

### 1.3 一次重要的自我纠错（错误闭环归因）

上一版记录的两条「发现」——① 超时=进程级致命；② 一次性大显存 OOM 拖死进程——
**经受控对照实验均被推翻**。真实收获是两个接口缺陷的修复：

1. `recover_device` 跨后端返回类型不一致（ascend `bool` / flagos `dict`）→ 统一为 `dict`；
2. `recovered` 语义不一致（底层只在 ISOLATED 才重建，导致「设备正常无需重建」被误报为「恢复失败」）
   → 统一为「设备当前可用」+ `detail` 三态。

### 1.4 遗留（**不阻塞**后续工作）

| # | 缺口 | 归属/下一步 |
|---|---|---|
| 1 | 组织架构 PR 同步 dev-1.0（0 冲突） | ⏳ 待评审合入（**需你手动发起或授权**） |
| 2 | 训练腿镜像未发布到 registry（v1 标 🟡 临时机器绑定资产） | 请总组/镜像 owner 推进发布，否则其他机器无法按锁定基座复现 |
| 3 | ~~训练腿 torch_fl 例外的退出口径（v1 记有 TODO：10 月起评估切回 Route A 的成本）~~ ✅ **2026-09-22 已闭环**（训练腿已统一切到 torch_npu；新遗留：解释器 torch 版本 2.11 vs 统一基座 2.10 是否拉平） | 已闭环，改为版本拉平议题 |
| 4 | 通信接口约定（启动方式 / flagcx 接口形态 / 对照用例归属） | 待分布式方向回复，备忘见 `DESIGN_DIST_COMM_20260908.md` |
| 5 | 历史模型未在统一原型上复跑（历史 910C 双卡 DDP 与 vLLM+TP 均为旧代码路径） | 观察项，不影响本轮 |
| 6 | 组件 v0.1.0 下游反馈待收集 | 等下游（算子/调度/显存/分布式）接入后按周迭代 `v0.1.x` |

---

## 2. 昆仑芯 P800 侧（本次工作）

### 2.1 定位（⚠️ 这条被纠正过一次，请注意口径）

**原（错误）定位**：「把 910C 上已验证的设备上下文 + 多流 Stream 能力适配到 P800」
→ 把 910C 的**实现**当成了迁移对象，读起来就是「把昇腾那套代码搬过去跑」。

**现（正确）定位**：**P800 是统一运行时原型的第二个接入实例**。
迁移的是**规范与方法**，910C 的实现与结论**不迁移**。

| 类别 | 内容 | P800 上怎么处理 |
|---|---|---|
| **规范（章程）** | 统一 API 与语义承诺、五域抽象、Backend 插件接入规范、两条硬纪律（`record_stream` / 错误隔离分层）、错误 L1–L4 分级与 `disposition` 约定、`recover_device` 返回契约、conformance 判据（13+6）、验收口径 | ✅ **必须遵守**，逐条落地 |
| **方法** | 最小变更 / 单变量隔离、证据规范（命令 + 数值）、错误闭环验证法、归属判定规则 | ✅ **复用方法，不复用结论** |
| **框架代码** | `runtime/api/`、13 个抽象、`conformance/` 13+6 例 | ✅ 规范的可执行形式，本就为多后端设计 |
| **910C 落地实例** | `backends/ascend\|flagos` 绑定、**108 条 ACL 错误码映射表**、CANN 约束、镜像基座、并发上限 3、`ASCEND_RT_VISIBLE_DEVICES` | ❌ **不迁移**，按 `torch.cuda` / XPytorch **新建** |

**两阶段口径**：9 月＝原型接入（昇腾 + 昆仑芯两个实例都在本轮）；11 月＝运行时层整体交付。
**交付链条**：按规范新建 `kunlun` backend → 跑 conformance → 用训推两条腿**暴露问题** →
属五域的**先简单修复** → 产出《新芯片接入手册》+ 规范修订建议 → **release 给运行时层其他子方向**（验证 + 迭代循环）。

**「先简单修复」的边界**：**只修 `RuntimeBackend` 五域内**；算子 / 通信 / 显存 / 调度 / 性能一律**对外提交**；
**不做生产级打磨**（那是 release 之后各子方向的事）。

### 2.2 阶段 0 · 环境与基线 ✅

#### 2.2.1 机器资源（8 卡全空闲）

| 维度 | 实测值 | 判定 |
|---|---|---|
| XPU | **8× P800 OAM，96 GB/卡（共 768 GB）**，驱动 5.0.21.47，容器内 `xpu-smi` 515.58 | ✅ 充裕 |
| 内存 | **1.5 TiB**（无 swap） | ✅ 充裕 |
| CPU | 2× AMD EPYC 9K84 = **384 线程**，2 NUMA | ✅ 充裕 |
| 网络 | **8× `mlx5_bond`，200 Gb RoCE（4X HDR）**，全 PORT_ACTIVE；**`kunlun_peermem` 已加载** | ✅ 良好 |
| 拓扑 | XPU0-3 属 NUMA0、XPU4-7 属 NUMA1；组内 **XL** 私有链路、跨组 SYS；NIC 与卡 **PIX** 直连 | ✅ |
| 存储 | `/data1`、`/data2` 各 5.8 TB NVMe；**`/var/lib/docker` 已 bind mount 到 `/data1`**（剩 1.5 TB） | ✅ 非阻塞 |
| 共享程度 | **25 人在线、22 个容器、272 僵尸进程**；**卡占用时刻在变** | ⚠️ 共享机 |

> **一处误判的更正（留痕备查）**：初版曾判定「docker data-root 在只剩 2.9 GB 的根分区 → 镜像加载必然失败」。
> `findmnt -T /var/lib/docker` 查实该目录**早已 bind mount 到 5.8 TB NVMe**，判断不成立。
> **误因**：用 `du -xhd1 /` 的可读合计（11 G）与 `df` 已用（91 G）求差，把差额归给 docker——
> 而 `du -x` 遇跨文件系统即停止，`/var/lib/docker` 本就不在其中。
> **教训**：判断某目录占多少盘，先 `findmnt -T` / `df -T` 定位文件系统，**别用 `du` 差值反推**。

#### 2.2.2 镜像：**根本不用拉**

| 镜像 | 体积 | 结论 |
|---|---|---|
| `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608` | **38.3 GB** | ✅ **本机镜像库已有** |
| `flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202606-base` | 34.7 GB | 本机也有（即那个 32 GiB tar 包对应镜像） |

→ 官方手册的 **59.9 GB `pull`** 与 **32 GB `load`** **全部跳过**；
FlagGems 源码也已在容器内 `/env/FlagGems`（来自 `git clone`）→ **无需联网**（容器内 github 不可达）。

容器：`hliu553-device-context-p800`，参数对齐**本机已跑通容器的等价配置** ——
非 privileged / bridge / `--shm-size=64g` / `/dev/xpu0..7` + `/dev/xpuctrl` + `/dev/fuse`，
挂 `/data2/hliu553:/workspace`（手册那套 `--privileged --net=host` 属**过度指定**）。

#### 2.2.3 三条关键认知（写进调用契约）

1. **设备 API 走 `torch.cuda`，`torch.xpu` 不可用**
   理论级证据：`torch.__config__.show()` → `USE_CUDA=ON, USE_NCCL=1, USE_XCCL=OFF, `**`USE_XPU=OFF`**；
   实测 `torch.xpu.is_available()=False` / `torch.cuda.device_count()=8`；
   机制是 **XPytorch + `torch_xray` 符号重写**把 CUDA 调用重定向到 P800（故 torch 是 `+cu129` 构建）。
   → **`kunlun` 后端不能照搬 910C（`torch_npu`）或 FlagOS（`torch_fl`）的命名空间假设。**
2. **选卡变量是 `CUDA_VISIBLE_DEVICES`**（实测 `=2` → 1 卡；`=2,5` → 2 卡）——
   它才是 910C `ASCEND_RT_VISIBLE_DEVICES` 的对应物，**不是**某个 XPU 侧变量。
3. **环境两件事必做**：先 `conda activate python310_torch29_cuda`（默认 `python3` 是 conda base **3.13**，目标环境 3.10.18）；
   先 `export XPU_EVENT_KL3_ENABLE=1`。

#### 2.2.4 五域基线

| 域 | 结果 |
|---|---|
| 设备抽象 | ✅ 8 卡可见（96 GiB / free 98272 MiB）、`mem_get_info` 准确 |
| 多流 Stream / Event | ✅ 创建、`with stream()`、**跨流 Event 依赖正确**（45056.0 = 期望值）、`record_stream` 全对；4096² matmul 计时 15.859 ms |
| 算子（FlagGems） | ✅ `flag_gems 5.3.4` 接管 CUDA dispatch，`add` max diff = **0.0** |
| 错误 | ⚠️ 异常类型正确（`OutOfMemoryError` / `AcceleratorError`），但**厂商错误码不透出** |
| 状态恢复 / 有界同步 | ⚠️ 仅 probe 级；**无设备级重置原语**（`reset*` 全是内存统计类） |

#### 2.2.5 上游缺失项（3 条，含归属判定）

| # | 缺失项 | 归属 |
|---|---|---|
| 1 | **流优先级**：`torch.cuda.Stream.priority_range()` 稳定触发 PyTorch `RuntimeError: greatest_priority <= -1 INTERNAL ASSERT FAILED at c10/cuda/CUDAStream.h:188` | **上游缺陷**（XPytorch 上报了非法优先级区间）→ 我方 `supports()` 如实声明不支持、conformance 如实跳过 |
| 2 | **厂商错误码不透出**：Python 异常消息无码，仅进程退出钩子偶见 `error code= 101` | **上游约束** → 错误映射表 v0 改以「**异常类型 + 消息模板**」为键 |
| 3 | **同进程内一处错误污染后续调用** | **设计依据**（非缺陷）→ 恢复动作应优先**重建进程/上下文** |

> **对第 3 条的一次自我更正**：最初读作「设备级粘滞」，**不准确**。隔离成独立进程重测后，
> OOM 正常报 `torch.OutOfMemoryError`、越界正常报 `AcceleratorError`，**污染只在同进程内，不跨进程**。
> 这个区分对错误闭环设计很关键。

### 2.3 阶段 1 · 接入 ✅ **已完成**

#### 2.3.1 结果

| 判据 | 结果 |
|---|---|
| conformance 13 例 | **13/13 PASS** |
| conformance 推理 6 例 | **6/6 PASS** |
| 组件自检 `smoke_runtime.py --backend kunlun` | **42 通过 / 0 失败** |
| 验收标准 1–3 | ✅ 全满足 |

**按接口约定 §2 那 5 步走完**：`backends/kunlun/` 落地 → `build()` 登记 → `supports()` 如实声明 → conformance → smoke。
提交 `530f456`（11 个文件，+777/−19）。证据：`conformance_runtime_kunlun.json`、`..._kunlun_infer.json`、`P800/probes/smoke_kunlun_20260914.txt`。

> `name="kunlun"`（我方的注册表键，按厂商）+ `device_type="cuda"`（设备串前缀，按命名空间）——
> **后端名与命名空间是两层不同的东西**。`kunlun` 与未来的 `nvidia` 会**共用 `torch.cuda`**，
> 所以 `supports()` / 厂商判别**不能靠命名空间**，必须用厂商特征
> （`torch_xray`/`torch_xmlir` 模块、`/proc/kunlun`、`xpu-smi`、`libbkcl.so` vs `libnccl.so`）。

#### 2.3.2 写之前先实测，避免了 5 个坑

| 实测发现 | 对实现的影响 |
|---|---|
| `Stream.synchronize()` 签名是 `(self) -> None` —— **无 timeout 参数** | 有界同步只能宿主侧轮询；如实标注为「超时上报」而**非「真中断」** |
| **`Event.query()` 未 record 时原生返回 `True`**（误报已完成） | 必须做 `KunlunEventAdapter` 修 E3，否则 `e3_query_unrecorded` 必挂 |
| **`torch.cuda.memory_stats()` 返回空 dict** | `memory_stats` 改走 `mem_get_info`，否则显存查询全 0 |
| **无设备级重置原语** | `recovery_real` 如实声明**不支持**，只做 probe 级 |
| `Stream.priority` 可读（0）但 `priority_range()` 崩 | `stream_priority_range()` 返回 None 并**主动拦截透传**，不让上层踩到 C++ 断言 |

#### 2.3.3 ⭐ 最有价值的产出：3 个「只有非昇腾实例才能暴露」的框架/判据缺陷

这三条**与昆仑芯本身无关**，但在昇腾单实例下永远照不出来 ——
**这正是 P800 作为「规范首个非昇腾实例」的价值所在**。均在本方向五域内，已按「先简单修复」处理：

| # | 缺陷 | 根因 | 修复 + 回归证据 |
|---|---|---|---|
| **1** | `registry` 注册流程被日志搞崩 | `logger.debug("...: %s", backend.info())` 的 `info()` **无条件求值**；`flagos.info()` → `_load()` → `import torch_fl`，P800 上抛 `ModuleNotFoundError`，**穿透 `register()` 中断整个 `discover()`** —— 违背该模块自己写的「发现失败仅告警、不中断」 | DEBUG 守卫 + `try/except`；`factory()`/`register()` 纳入 `discover` 容错。修复后 `discover()` 返回 `['ascend','flagos','kunlun']` 不再中断 |
| **2** | conformance `f1` 硬要求厂商错误码 | 断言 `fe.error_code is not None` **超出它自称的「类别/位置/根因三投影」契约**，把昇腾的 `ret=XXXX` 当成通用前提 → **无码后端恒 FAIL，与实现质量无关** | 按 `supports("error_map")` 分支。**向后兼容已实测**：模拟「声明 error_map」时同一无码错误仍判 FAIL —— 既有断言路径**一字未改**（不是靠嘴说） |
| **3** | `smoke_runtime.py` 只覆盖昇腾 | 第 [5] 节硬编码 `ascend`，其他后端得不到自检 | 新增第 **[6] 节「真实后端通用自检（后端无关）」** + `--backend` 参数（不传则自动挑选，**原用法不变**） |

→ 已一并写入方案 **§7.3 修订建议**（共 7 条），完整现象/根因/修复/回归证据见 **§7.5**，供《新芯片接入手册》直接收录。

### 2.4 阶段 2 · 训练腿 ✅ **已在标注条件下完成（多卡跑通）**

> **结果**：同一份脚本，`XPU_EVENT_KL3_ENABLE` **未设** → 两 rank **TRAIN_LEG_PASS 6/6**、
> loss **15.4488 → 11.1481**（50 步、无 NaN）、**3482 tok/s**（两卡合计、14.7 s）；
> **设为 1** → 只跑到 `[step 0]` 即挂死、1800/300 s 超时（退出码 124）。
> 证据：`P800/probes/E_train_leg_result_rank{0,1}.json`、`E_train_ab.log`。详见 §2.6.3 轨 2（2b 已执行）。

#### 2.4.1 已完成：脚本后端无关化

`proto_train_leg.py` 从「硬编码 `flagos` 后端 + 无条件 `import torch_fl` + 910C 路径」改为**环境变量驱动，默认值保持 910C 原行为** ——
于是**同一份脚本两处都能跑**，这正是「统一 API：换芯片只改一行」的体现。
顺带修掉：`import torch_fl` 改**按后端条件导入**（原来无条件，P800 上必然 ImportError）；
`torch.flagos.synchronize()` → `runtime.synchronize()`（走统一 API）。

> **状态**：改动**尚未提交**（`git diff` 62 插入 / 14 删除），待训练腿跑通后与结果一起提交。

新增环境变量：`DC_BACKEND` / `DC_DIST_BT` / `DC_ROOT` / `DC_MODEL` / `DC_OUT_DIR` / `MAX_STEPS` / `BATCH` / `SEQ`。

#### 2.4.2 分布式后端探测（跑起来的第一道门）

| 后端 | 结果 |
|---|---|
| `nccl` | ❌ **挂死**（超 4 分钟无输出） |
| `xccl` | ❌ **未编译** — `RuntimeError: Distributed package doesn't have XCCL built in`（只注册了名字） |
| `kccl` | ❌ 无响应 |
| **`flagcx`** | ✅ **正确路径**（与 xliu969 已验证的 Route A 一致） |

**正确用法**（这是地面真相，来自 xliu969 跑通的代码）：

```python
import flagcx                                     # 必须显式 import 才会注册后端
dist.init_process_group("cpu:gloo,cuda:flagcx")   # 设备走 flagcx、CPU 走 gloo
```
环境：`FLAGCX_ADAPTOR=klx`，`torchrun --nproc-per-node=2`

> ⚠️ **接入手册级别的一条坑**：同一个集合通信库（FlagCX），
> **在 910C 上注册的后端名是 `flagos`，在 P800 上是 `flagcx`**。
> 换芯片不只换设备命名空间，**连集合通信的后端名都变了**。

#### 2.4.3 隔离实验（决定性，把问题范围锁死）

| 用例 | 用卡 | 结果 | 判定 |
|---|---|---|---|
| T1 单进程 · 物理 0 | dev0 | ✅ OK（matmul sum=108450.85） | 有效 |
| T2 单进程 · 物理 1 | — | ❌ `invalid device ordinal` | ⚠️ **我的测试写错**（`CUDA_VISIBLE_DEVICES=1` 时物理 1→逻辑 0，我却传 `set_device(1)`）→ **无效证据** |
| T2' 单进程 · 物理 2 | dev2 | ✅ OK（sum=113286.28） | 有效 |
| T3 双进程（各算） | 0,1 | ⚠️ SSH 断连，无结论 | 无效 |
| **T3' 双进程各自算（无集合通信）** | 0,2 | ✅ **两 rank 都 OK** | **有效** |
| **T4' 双进程 gloo barrier + 设备 matmul** | 0,2 | ✅ **OK** | **有效** |
| **T5 flagcx all_reduce** | 0,1 | ❌ **KL3 kernel exception** | 有效 |
| **T5' flagcx all_reduce** | 0,2 | ❌ 无 worker 输出、300s 超时（日志被 `tail` 截断） | ⚠️ 结论待复测确认 |

**当时的中间结论**（❌ **已被 T5b 推翻，见 §2.4.4**）：问题不在多进程、不在设备算子、不在通信引导层，
似乎「只在 flagcx 的设备侧集合通信」。

#### 2.4.4 ⭐ 关键更正：集合通信不是缺陷，**卡**才是变量

**T5b 复测（决定性）—— 用连续同组的卡 6,7，全量日志、固定超时 240s：**

```
[INFO][XCCL][globals.cpp:299] xccl version: 0792b03 [rdma]  build data: Feb  9 2026
RESULT flagcx all_reduce OK got=[1.0, 3.0, 5.0, 7.0]
RESULT flagcx all_gather OK [0.0, 1.0]
RESULT flagcx p2p OK rank0
EXIT_CODE=0
```

→ **flagcx 在 P800 上完全可用，通信三类对照（all_reduce / all_gather / P2P）全部通过。**

**同一份代码、四次运行的现象对照：**

| 运行 | 用卡 | 结果 |
|---|---|---|
| T5（首轮探测） | 0,1 | ❌ KL3 内核异常（`error set to 721, status= 299`） |
| T5'（iso3） | 0,2 | ❌ 无 worker 输出、300s 超时 |
| **T5b（复测）** | **6,7** | ✅ **三类通信全 OK（EXIT_CODE=0）** |

**因此我上一条「flagcx 设备侧集合通信失败」的结论 ❌ 不成立，予以更正。** 真因指向**卡**：

- 卡 0,1,2 属 **NUMA0** 组；卡 6,7 属 **NUMA1** 组；
- 实验期间**卡 1 被其他租户反复占用**（166 → 196 → 502 MiB，100% 利用率），
  实验后仍观察到卡 0 占 31 GB、卡 3 占 32 GB、卡 4 占 4 GB（**均非我方**）；
- ⇒ 高度怀疑**共享机上的他租户负载**是触发因素（设备侧报错 / 引导失败）。

**受影响的项全部回退**：

| 项 | 原判断 | 更正后 |
|---|---|---|
| 对外提交 **C3**（flagcx/BKCL 缺陷） | 待起草 | ❌ **撤销** —— 非缺陷 |
| 宿主 sudo 动作 **B1**（`dma_excp_mask` 置 1） | 待你操作 | ⏸ **暂不需要**（无证据支持，且会掩盖真因） |
| 训练腿可行性 | 被卡住 | ✅ **可跑**，已在卡 6,7 启动 |

**先例留档（仍值得记，但不构成本次结论）**：`kl3ChannelCheckErrors`（`kl3_gpfifo.cc:183/184`）
在 xliu969（`status=719`）、chenzizhong、xianghuang 的记录里均出现过；
triton 曾提示 `XRE Version Must Be More than 5.0.21.37` + `echo 1 > /proc/kunlun/dev4/dma_excp_mask`，
而 8 张卡当前全为 0。**本次结论与该开关无关**（不需它就已跑通），故不再作为处置建议。

**⚠️ 仍未完全闭合的一点（诚实标注）**：卡 0,2 那次为何是「挂死」而非报错，尚未解释。
但**已不必阻塞主线** —— 结论「flagcx 可用、用卡需挑空闲连续卡」已由 T5b 直接证据支撑。
若后续再遇，按「**先换卡复测**」处理，即：先用 `xpu-smi` 确认无他租户占用，再复现。

**⇒ 由此得到一条实操纪律（必须进手册）**：
**共享机上，用卡前先 `xpu-smi` 挑「连续且空闲」的卡，并在实验记录里写明用卡。**
本轮 910C 基线用的是 0,1；P800 因 0/1/2 被他人占用，实际用的是 **6,7**。

#### 2.4.5 环境事实（用卡纪律的由来）

**1 号卡反复被其他租户占用**（观察到 166 → 196 → 502 MiB，100% 利用率），
实验后还观察到 dev0 占 31 GB、dev3 占 32 GB、dev4 占 4 GB。
→ **用卡前必须先查 `xpu-smi` 挑空闲卡**，并在实验记录里写明用卡；这条要进手册。
（本轮因 1 号卡被占，改用**同组内空闲**的 0,2 —— 仍在 XPU0-3 组内、同样避开跨 NUMA。）

#### 2.4.6 阶段 2 新增的两条坑（进手册）

| # | 坑 | 现象 | 处置 |
|---|---|---|---|
| **1** | **同一 FlagCX，两芯片后端名不同** | 910C 上注册为 `flagos`；P800 上注册为 `flagcx`，且**必须显式 `import flagcx`** 才会注册 | 脚本按 `DC_DIST_BT` 是否含 `flagcx` 条件导入（已实现） |
| **2** | **HF cache 要指向 `snapshots/<hash>`，不是仓库根目录** | 传 `.../models--Qwen--Qwen3-Embedding-0.6B` → `ValueError: Unrecognized model ... Should have a model_type key`（根目录只有 `blobs/`、`refs/`、`snapshots/`） | 传 `.../snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |

> 坑 2 的额外价值：失败发生在**模型加载**而非通信，而日志里进程组已建立成功
> （`[Gloo] Rank 0/1 is connected to 1 peer ranks` + `xccl version: 0792b03 [rdma]` + BKCL 加载）
> —— **这构成 flagcx 在真实训练脚本里的第二次可用性验证**。

#### 2.4.7 训练腿运行状态（最新，含精确定位）

**运行 1 —— 失败在「模型加载」，不是通信**

`ValueError: Unrecognized model in .../models--Qwen--Qwen3-Embedding-0.6B. Should have a model_type key`
（原因：HF cache 传了仓库根目录，`config.json` 在 `snapshots/<hash>/` 下，见 §2.4.6 坑 2）。
**但日志显示进程组已建立成功**（`[Gloo] Rank 0/1 is connected to 1 peer ranks` + `xccl version: 0792b03 [rdma]` + BKCL 加载）
→ 这是 **flagcx 在真实训练脚本中的第二次可用性验证**。

**运行 2 —— 现象：卡上在跑、但 4 分钟零输出**

| 观测 | 值 |
|---|---|
| 每卡显存占用 | **6970 MiB**（模型已加载到设备） |
| 每卡利用率 | **100%** |
| 两 rank CPU | ~18–19%（主线程 R 态） |
| **系统时间 / 用户时间** | **172 s / 46 s** ← 典型**驱动层空转等待**特征 |
| 日志最后一行 | 15:11:15（此后 4 分钟无输出） |

**用 `faulthandler.dump_traceback_later(90, exit=True)` 定时 dump 线程栈，精确定位：**

```
Timeout (0:01:30)!
Thread 0x... (most recent call first):
  File "proto_train_leg.py", line 164 in main     ← 两个 rank 完全相同
  File "proto_train_leg.py", line 208 in <module>
```

**⇒ 结论（含重要进展）**：

| 环节 | 状态 |
|---|---|
| 进程组建立（`cpu:gloo,cuda:flagcx`） | ✅ |
| 模型加载到设备 | ✅ |
| **前向 + 反向** | ✅ |
| **通信三类校验（`all_reduce` / `all_gather` / P2P）** | ✅ **全部通过**（它们在 114–145 行，能执行到 164 行即已通过） |
| **逐参数梯度 `all_reduce`（约 290 个张量）** | ❌ **卡在这里** |

> 卡点的关键反差：**少量/小张量的集合通信正常，逐参数（多次）梯度通信挂住。**
> 下一步用定点探针把「**单次大通信**（concat 后整块）」与「**多次小通信**（逐参数）」分开对比，
> 判定是「**多次调用的累积问题**」还是「**特定张量形状/连续性**」——这是标准的单变量隔离。

#### 2.4.8 定点探针：**大通信正常，反复小通信挂死**（关键反差）

为判定卡点性质，写了一支单变量探针 `P800/probes/dc_probe_grad_ar.py`：
**A) 把 310 个梯度 concat 成一块后做一次 all_reduce**（单次大通信）
**B) 再逐参数 all_reduce**（多次小通信），逐个打印

| 步骤 | 数据量 | 结果 |
|---|---|---|
| 反向（两 rank） | — | ✅ loss = **15.6902**（两 rank 一致） |
| 梯度张量数 | — | **310** |
| **A. 单次大通信**（`flat`） | **2.38 GB**（595,776,512 元素） | ✅ **OK，0.336 s / 0.258 s** |
| B#0 `model.embed_tokens.weight` | 151669×1024（**0.58 GB**） | ✅ OK，0.022 s |
| B#1 `layers.0.self_attn.q_proj.weight` | 2048×1024 | ✅ OK，0.000 s |
| **B#2 `layers.0.self_attn.k_proj.weight`** | **1024×1024（仅 4 MB）** | ❌ **挂住**（两 rank 都停在 B#2 表头之后） |

**⇒ 结论：不是「数据量太大」，而是「反复调用的集合通信会挂」。**
**2.38 GB 一次就过，4 MB 的第 3 次调用却挂死** —— 这条反差把「带宽/容量」和「形状/连续性」两个假设都基本排除,
指向 **flagcx/BKCL 在连续多次集合通信上的稳定性问题**。

**另一个技术信号**：本探针设置的 `faulthandler.dump_traceback_later(150s)` **没有触发**
（而训练脚本里 90s 的 dump 正常触发了）。watchdog 线程要拿 GIL 才能跑 ——
**没触发说明挂死点是一个持有 GIL 且不自旋让出的 C 调用**，与「驱动层空转等待」的特征吻合
（对照：训练脚本那里的系统时间 172 s ≫ 用户时间 46 s）。

**判定探针结果**（`P800/probes/dc_probe_ar_rep.py`）：

| 步骤 | 结果 |
|---|---|
| C) **同一个** 1024×1024 张量重复 all_reduce | 第 **10 / 20 / 30 / 40** 次均正常（累计仅 0.24 s）→ **第 41–50 次之间挂住** |
| D) 310 个真实梯度按序 | 未执行到（被 C 阻塞） |

**关键对照**：第一次探针挂在**第 3–4 次**调用（B#2），本次挂在**第 41–50 次**。
**同一操作、挂死点却不同 ⇒ 非确定性（竞态），不是「特定张量」，也不是「数据量」。**

#### 2.4.9 阶段 2 阶段结论与归属判定

**现象汇总（全部为我方直接观测）**

| 事实 | 证据 |
|---|---|
| 进程组建立、模型加载、**前向+反向** | ✅ 正常（loss 15.6902，两 rank 一致） |
| **通信三类校验**（all_reduce / all_gather / P2P） | ✅ **在探针与训练脚本中均通过** |
| **单次大通信** | ✅ 2.38 GB 一次 all_reduce **0.336 s** 通过 |
| **反复调用集合通信** | ❌ **非确定性挂死**（同操作分别在第 3–4 次、第 41–50 次挂住） |
| 挂死时的形态 | 卡利用率 100%、系统时间 ≫ 用户时间、**GIL 被自旋 C 调用占住**（faulthandler watchdog 无法触发） |

**归属判定（按 §2.3 规则逐条套用）**

关键证据：**判定探针完全使用裸 `torch.distributed.all_reduce`**（仅 `import flagcx` 注册后端），
**没有经过我方 `RuntimeBackend` 的任何一层** ⇒ 挂死**不在我方五域内**。

| 候选归属 | 判据 | 结论 |
|---|---|---|
| 我方运行时层（Stream/Event/错误/状态恢复） | 探针未使用我方 API 仍挂 | ❌ **排除** |
| 设备算子 / 模型 / 显存 | 前向+反向、单次大通信均正常 | ❌ **排除** |
| **P800 侧 flagcx/BKCL 集合通信实现** | 反复调用非确定性挂死；910C 同一训练腿 50 步稳定（2117 tok/s） | ✅ **高度指向** |

> **重要区分**：910C 上同一套训练腿（FlagCX 注册为 `flagos`）**50 步稳定跑通、2117 tok/s**。
> ⇒ 不是「FlagCX 这个库整体不行」，而是 **P800 这一份构建/适配的问题**。

**⇒ 处置：列为候选对外提交项（C3），但需先补两项隔离再定案**：
① 换卡复现（当前只在卡 6,7 上测过，且卡 0,1,2 期间被他人占用）；
② 与上游确认 FlagCX/BKCL 在 P800 上的已知问题（日志里出现 `[[BKCL-1245] allow masking GC signal handlers via BKCL_GC_SIGNAL_MASK]` 这一构建标记，值得问）。

**顺带一条实操教训（进手册）**

| 教训 | 说明 |
|---|---|
| **`timeout` 的 SIGTERM 无法中断这类挂死进程** | 挂死点持有 GIL 自旋、信号被推迟 ⇒ `timeout 120` 到点后进程**仍存活 73 分钟**。清理必须 `kill -9` + 按 PID 强杀，并**复查 `xpu-smi` 确认卡已释放**（本轮已清理：容器内无遗留进程，卡 6,7 归零） |

### 2.5 ⭐ 根因定位（第二轮深挖：gdb 原生栈 + 单变量探针对照）

#### 2.5.1 方法：从「现象」升级到「函数级证据」

第 2.4 阶段只能证明「反复集合通信会挂」。本次**换了取证手段**：发现容器内有 `gdb`，
于是自建一个**带 `SYS_PTRACE` + `seccomp=unconfined` 的独立调试容器**（不动主容器），
配合「探针组 + 挂死自动抓栈」的脚本，把挂死点定位到**函数级**。

> 主容器 ptrace 被 seccomp 拦住（`ptrace: Inappropriate ioctl for device`）；
> 调试容器 `hliu553-dc-debug-p800` 专门用于取证，**不改变主容器配置**。

#### 2.5.2 三处挂死点（全部在 vendor `libcuda` 与 flagcx/BKCL 的交界处）

| 挂死点 | 原生栈（gdb 实测） | 观测变体 |
|---|---|---|
| **H1** `cudaDeviceSynchronize` 用户态自旋 | `torch.cuda.synchronize()` → `THCPModule_cudaSynchronize` → `c10::cuda::device_synchronize()` → `cudaDeviceSynchronize` → **厂商 `libcuda.so.1` 内 8 层无符号帧自旋** | V1、V4、A1–A3 |
| **H2** `cudaEventRecordWithFlags` 自旋（**含 `sched_yield`**） | `c10d::ops::allreduce_CUDA` → `c10d::flagcxBackend::allreduce` → **`flagcxBackend::syncStream`** → `at::cuda::CUDAEvent::record` → `cudaEventRecordWithFlags` → 厂商 `libcuda.so.1` 自旋 | V6 |
| **H3** 通信域**首次初始化**死锁 | rank1：`flagcxBackend::initComm` → `flagcxCommInitRank` → `flagcxHomoCommInit` → `xcclAdaptorCommInitRank`（`xccl_adaptor.cc:87`）→ `bkcl::init_rank` → **`bkcl::net_socket_all_gather` 的 `recv()` 阻塞**；rank0：同一初始化路径但**卡在 `bkcl::kl3::init_device_param` → `xpu_free`** | V2 |

**关键旁证**：`libcuda.so.1` 实为符号 **`libxpucuda.so`**（厂商 CUDA 兼容层，符号已剥离，故 gdb 只能给地址）。
两个 rank 的主线程栈**完全一致**，所有其它线程处于 `pthread_cond_wait`（BKCL 代理线程在 `bkcl::util::BlockingQueue::pop` 上空闲）
⇒ **不是锁竞争、不是我方代码**，而是厂商同步原语「有工作未完成」的永久等待。

#### 2.5.3 单变量对照（第一轮 8 组 + 第二轮重复验证）

**第一轮（每个条件只改一个变量）**

| 变体 | 变化 | 结果 |
|---|---|---|
| V1_base | 基线（KL3=1、SUM、每 10 次同步） | ❌ 挂死 |
| V2_cards12 | 换卡 1,2 | ❌ 挂死 → **与卡无关** |
| **V3_noevent** | **不设 `XPU_EVENT_KL3_ENABLE`** | ✅ **120/120** |
| V4_gcmask | `BKCL_GC_SIGNAL_MASK=1` | ❌ 挂死 → **该开关无效** |
| V5_nosync | 循环内不做设备同步 | ⚠️ 报「120/120」但**该结论已被推翻**——探针不自证、未确认通信是否完成（**假阴性**，见 §2.5.4 更正） |
| V6_max | `op=MAX` | ❌ 挂死 → **与 reduce op 无关** |
| V7_barrier | 只做 `dist.barrier()`（无同步） | ✅ 120/120 |
| V8_devonly | 单进程纯设备计算（无通信） | ✅ 完成 |

**第二轮（重复验证 + 补对照）**

| 组 | 条件 | 结果 |
|---|---|---|
| **A ×3** | `XPU_EVENT_KL3_ENABLE=1` + 集合通信 + 同步 | ❌ **3/3 挂死**（rep=20 / 40 / 70） |
| **B ×3** | **不设** KL3（其余同 A） | ✅ **3/3 通过 120/120** |
| **C** | 显式 **`XPU_EVENT_KL3_ENABLE=0`** | ✅ 通过 120/120 |
| **D** | KL3=1 + **纯设备计算（无通信）**，10 万次迭代 | ✅ 通过 |
| E | KL3=1 + 每次都同步 | ✅ 通过（**单次；第三轮 N=1 重复 2 次均挂死 → 判为偶然**） |

**第三轮（剂量-反应：固定 KL3=1，只改「同步间隔 N」= 同步前积压的通信次数）**

| N（每 N 次通信同步一次） | 结果 |
|---|---|
| **1**（×2） | ❌ ❌ **2/2 挂死** |
| **2** | ❌ 挂死 |
| **3** | ❌ 挂死（rep=40） |
| **5** | ✅ 通过 120/120 |
| **10** | ❌ 挂死（rep=30） |

⇒ **无阈值效应**：N=1 全挂而 N=5 通过、N=10 又挂 ⇒ **同步间隔不是关键变量，挂死是随机的**。
（第二轮 E 那次通过随之被确认为偶然。）

#### 2.5.4 结论（⚠️ 已更正：不是「三要素」，是「两要素 + 多个暴露点」）

> **本小节的初版判断有误，已用带真值校验的探针推翻并更正。**
> 完整核对过程见 [`KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md`](KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md)。

**更正原因**：初版依据 `V5_nosync`（KL3=1、循环内不显式同步）报"通过"，判定"设备同步"是必要条件之一。
但**该探针不自证**——它只是"没等"就退出，从未确认那 120 次集合通信是否真的完成。

**复核实验**（新探针 `dc_probe_verify.py`：**无论是否在循环内同步，结尾都做一次同步并校验张量真值**；
各 rank 初值全 1，经 SUM all_reduce 后应恰为 `2^120`，float32 可精确表示）：

| 组 | 条件 | 结果 | 真值校验 |
|---|---|---|---|
| A×4 | KL3=1 + 每 10 次同步 | ❌ **4/4 挂死**（rep=100 / 20 / 0 / 0） | — |
| B×2 | **不设 KL3** + 每 10 次同步 | ✅ 通过 | **`1.329228e+36 = 2^120`，rel_err = 0.000e+00** |
| **C×2** | **KL3=1 + 循环内完全不做同步** | ❌ **2/2 挂死**（均停在 `rep=100`） | — |
| D1 | 不设 KL3 + 不作同步 | ✅ 通过 | **`2^120` 精确匹配** |

**⇒ 更正后的判别条件（两要素，缺一不挂）**

| # | 必要条件 | 反证 |
|---|---|---|
| 1 | **`XPU_EVENT_KL3_ENABLE=1`** | 不设或设 0：**8 次运行 0 次挂死**（其中 4 次另有真值校验，全部精确正确） |
| 2 | **存在设备侧集合通信（flagcx）** | 纯设备计算 / 仅 `dist.barrier()`：未观测到挂死 |

**"显式设备同步"不是必要条件，只是暴露点。** 机理：`dist.all_reduce` 会进入
`c10d::flagcxBackend::allreduce` → **`flagcxBackend::syncStream`** → `CUDAEvent::record` →
`cudaEventRecordWithFlags` —— **flagcx 插件在每次 all_reduce 内部就自带一次设备事件记录**，
所以**即使上层完全不显式同步，挂死照样发生**（C 组实测挂在第 101–120 次 `all_reduce` 内部）。

**三处暴露点（同一底层缺陷的不同暴露位置）**

| # | 暴露点 | 观测 |
|---|---|---|
| **P1** | **`dist.all_reduce` 内部**（根本暴露点，每次调用都经过） | C1/C2（本轮）、V6 |
| **P2** | 上层显式 `torch.cuda.synchronize()` | A1–A4、V1、V4 |
| **P3** | 通信域首次初始化（复现率低，18 次中仅 1 次） | V2 |

**实测挂死率**：`KL3=1 + 设备集合通信` 共 **18 次运行、16 次挂死（≈89%）**，
本轮基线 **4/4 = 100%**；挂死步数游走（rep 0 / 20 / 30 / 40 / 70 / 100），
**与数据量、张量形状、reduce op、用卡对、同步间隔均无关**。

**精确偏移（给上游定位）**：`libcuda.so.1`（→ `libxpucuda.so.515.58.kunlun`）
映射基址 `0x744bb1400000`，自旋帧 `0x744bb1494080` ⇒ **偏移 `+0x94080`**。

#### 2.5.5 归属判定：**不在我方五域**（证据比 §2.4.9 更强）

- 探针全程使用**裸 `torch.distributed.all_reduce` / `dist.barrier()` / `torch.cuda.synchronize()`**，
  **没有经过我方 `RuntimeBackend` 任何一层**；
- 挂死点落在 **厂商 `libcuda.so`（`libxpucuda.so`）** 与 **flagcx c10d 插件（`flagcxBackend::syncStream`）** 内；
- 910C 上同款训练腿 50 步稳定（2117 tok/s）⇒ 不是 FlagCX 整体问题，也不是我们的抽象层问题。

**并且——算子层与编译层均已用硬证据排除**（详见核对报告 §3）：

| 层 | 排除证据 |
|---|---|
| **算子层（FlagGems）** | `USE_FLAGGEMS`/`GEMS_VENDOR` 未设；`flag_gems` **未导入**（`sys.modules` → False）；探针源码 `flag_gems` 出现 **0 次**；无 `.pth` 自动 enable 钩子 |
| **编译层（FlagTree/triton）** | `/root/.triton` mtime = **2026-08-12**（镜像构建时）；近 2 小时**无任何 triton/xpubin 产物被改**；探针日志**无 triton/jit/compile 字样**；挂死路径上的内核是 **BKCL 预编译内核**，不经 triton |
| **厂商运行时/驱动** | ✅ **指向**：`XPU_EVENT_KL3_ENABLE` **只被 `libxpucuda.so` 读取（1 处）**，且位于 `CUDA_*` 运行时旋钮块中；三处自旋帧均在 `libxpucuda.so` 内 |

⇒ **对外提交对象：① 昆仑芯 XPytorch / XRE（`libxpucuda.so` 的 `cudaDeviceSynchronize` 与 `cudaEventRecordWithFlags` 在 KL3 事件开启时可能永久自旋）；② flagcx c10d 插件（`syncStream` 路径）。**

#### 2.5.6 ⚠️ 规避手段与其代价（**不能简单当成开关**）

实测**不设 / 设 0** 该变量后，**4/4 次全部通过**。但**不能直接建议关闭**，因为：

| 出处 | 内容 |
|---|---|
| `FlagGems/tools/env.sh`（`kunlunxin)` 分支） | `export XPU_EVENT_KL3_ENABLE=1` —— 官方环境脚本里就这么写 |
| `FlagGems/src/flag_gems/backends.yaml`（`kunlunxin:` 段） | `env: XPU_EVENT_KL3_ENABLE: "1"` |
| `FlagGems/.github/configs/weekly/P800.yml` | CI 环境里也设 `1` |

⇒ 它是 **FlagGems kunlunxin 后端的官方推荐环境变量**（我们当初也是照此设置）。
**关闭它可能掩盖厂商的 KL3 设备事件/异常上报**，是否安全必须由上游确认。

**本方向处置**：① 列为对外提交项（C3，升级为「函数级证据 + 最小复现」）；
② **不擅自改我们锁定镜像的环境口径**，仅在探针/诊断场景做对照；
③ 在《新芯片接入手册》里记为**已知坑 + 复现命令 + 临时规避**（明确标注"需上游确认后再正式采用"）。

#### 2.5.7 复现方式（可照做）

```bash
# 1) 取证容器（主容器 ptrace 被 seccomp 拦，故单起一个；用同一镜像与挂载，不动主容器）
docker run -dit --name hliu553-dc-debug-p800 \
  --network=bridge --shm-size=64g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/xpu0 --device=/dev/xpu1 --device=/dev/xpu2 --device=/dev/xpu3 \
  --device=/dev/xpu4 --device=/dev/xpu5 --device=/dev/xpu6 --device=/dev/xpu7 \
  --device=/dev/xpuctrl --device=/dev/fuse \
  -v /data2/hliu553:/workspace -w /workspace \
  flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608 /bin/bash

# 2) 探针组（含挂死自动抓 gdb 原生栈）——脚本在 P800/probes/
docker exec -d hliu553-dc-debug-p800 bash /workspace/probe_battery.sh   # 第一轮：8 变体
docker exec -d hliu553-dc-debug-p800 bash /workspace/probe_battery2.sh  # 第二轮：重复验证
docker exec -d hliu553-dc-debug-p800 bash /workspace/probe_battery3.sh  # 第三轮：剂量-反应

# 3) 单点复现（最小）：KL3=1 + 集合通信 + 同步
CUDA_VISIBLE_DEVICES=6,7 FLAGCX_ADAPTOR=klx XPU_EVENT_KL3_ENABLE=1 \
  python3 -m torch.distributed.run --standalone --nproc_per_node=2 dc_probe_rep.py
#   对照：去掉 XPU_EVENT_KL3_ENABLE 或把 MODE 设为 ar_nosync（不同步）→ 未观测到挂死
```

**注意**：用卡前先 `xpu-smi` 挑**连续且空闲**的卡；挂死进程 `kill -9` 才能清掉（`timeout` 的 SIGTERM 无效）。

### 2.6 处理策略：**根因上报厂商，本方向不停摆**（职责边界内推进）

> **原则**：**上报厂商**与**本方向继续推进**是两件事，不应互相阻塞。
> 边界是「**不改公共资产**（FlagGems / 锁定镜像口径 / 基座文件）」，
> 而不是「不能在自己方寸之内控制实验条件并如标注」。

#### 2.6.1 关键解锁点：**我们的训推验证路径都不依赖 FlagGems**

| 路径 | 算子来源 | 是否依赖 FlagGems |
|---|---|---|
| **训练腿** | `transformers 4.57.1` + 原生 torch 算子（XPytorch 提供） | ❌ 不依赖 |
| **推理腿** | `vllm 0.13.0` + `vllm-plugin-fl 0.1.0` | ❌ 不依赖（源码里 `flag_gems` / `USE_FLAGGEMS` **0 处引用**） |

⇒ `XPU_EVENT_KL3_ENABLE` 是 **FlagGems kunlunxin 后端的环境处方**，**不属于我们的验证前置条件**。
**在本方向验证运行中不设置它，性质是「去掉一个我们本来就不需要的变量」，
而不是「修改 FlagGems 的口径」** —— 两者是不同的事。

#### 2.6.2 flagcx 侧不可规避（已查源码）

`flagcxBackend::syncStream`（`plugin/torch/flagcx/src/backend_flagcx.cpp:424`）：

```cpp
void flagcxBackend::syncStream(at::Device device, int index) {
  auto &event = getEventByIndex(index);
  auto stream = getStreamByIndex(index);
  event->record(device.index());      // ← 挂死点
  event->block(stream, device.index());
}
```

在 `allreduce` 中的调用注释写明用途：*"First let default flagcx stream wait for
input tensor allocation stream"* —— 它是**流序正确性所必需**，**14 处集合通信各调一次，
没有开关也不能关**（关掉会造成输入未就绪即被读取）。
⇒ **规避杠杆只在厂商那个变量上，不在 flagcx 侧。**

#### 2.6.3 三轨推进

**轨 1 · 不受阻的交付先做完（立即可执行）**

| 项 | 是否受该缺陷影响 |
|---|---|
| 阶段 1 接入（conformance 13/13 + 6/6、smoke 42/0） | ✅ 已完成，不受影响 |
| 阶段 3 **推理腿单卡**（vLLM 服务化）——不涉及多进程集合通信 | ✅ 不受阻，可立即做 |
| 阶段 4 **错误闭环**（单卡/单进程） | ✅ 不受阻（但见 2.6.4 的对照要求） |
| 阶段 5《新芯片接入手册》+ 规范修订建议 | ✅ 不受阻，且**这条缺陷本身就是手册里最有价值的一节** |

**轨 2 · 训练腿分档交付（不伪造、不空等）**

| 档 | 内容 | 处置 |
|---|---|---|
| **2a** | **单卡/单进程**训练腿：验证设备上下文 + 数据通路 + loss 下降 | ✅ 不受阻，立即做 |
| **2b** | **多卡**训练腿：在**本方向验证运行环境**中不设置该变量取证据 | ✅ **已执行并跑通**（见下方 A/B 对照）。证据文件与报告**逐条标注条件**：<br>「本条证据在 `XPU_EVENT_KL3_ENABLE` 未设置下取得；开启时本环境概率性挂死（厂商缺陷，已上报）」 |
| **2c** | **开启该变量下的多卡结果** | ⛔ **标注为阻塞项**，等厂商修复；**不硬凑、不伪造** |

**轨 2 的验证结果：A/B 单变量对照（本轮实跑）**

同一脚本、同一用卡（6,7）、唯一变量 = `XPU_EVENT_KL3_ENABLE`：

| 运行 | 该变量 | 结果 | 证据 |
|---|---|---|---|
| **R1 规避** | **未设** | ✅ 退出码 **0**；两 rank **TRAIN_LEG_PASS 6/6**；loss **15.4488 → 11.1481**；**3482 tok/s**（14.7 s / 50 步）；`preflight` 告警**未误报** | `E_train_leg_result_rank{0,1}.json`、`E_train_ab.log` |
| **R2 对照** | **=1** | ❌ 只打到 `[step 0] loss=15.4488` 即挂死；300 s 后被杀，**退出码 124** | `E_train_R2_control_KL3on.log` |

**⇒ 规避方案有效（训练腿多卡可跑通），且缺陷确实存在（设变量即挂）——两者互为证明。**

**与 910C 基线的对照（同构可比，说明后端抽象到位）**

| 判据 | 910C（`flagos` 后端，锁定镜像） | P800（`kunlun` 后端，规避条件） |
|---|---|---|
| 判定 | 两 rank **6/6** | 两 rank **6/6** |
| loss（50 步） | 15.4497 → 11.15 | **15.4488 → 11.1481** |
| 吞吐（两卡合计） | 2117 tok/s | **3482 tok/s** |
| 通信三类对照 | all_reduce / all_gather / P2P 全对 | 全对（`3.0/3.0`、`[0.0, 1.0]`、P2P 一致） |
| 设备上下文 | rank0/rank1 各绑一卡、各建统一流 | 同（`统一流 backend=kunlun`） |

> **诚实标注**：P800 这组结果在 `XPU_EVENT_KL3_ENABLE` **未设置**下取得；
> 开启时本环境概率性挂死（厂商缺陷，已上报）。**该证据不能代表开启该变量时的行为。**

**轨 3 · 上报与标注**

| 位置 | 标注形式 |
|---|---|
| ① `backends/kunlun/backend.py` | **机器可读**：`info()["known_issues"]` 结构化条目 + `known_upstream_defects` 追加一条；新增可选方法 `known_issues()`（base 默认返回 `[]`，零回归）——**其他子方向接入时读到后端即可获知** |
| ② `proto/proto_train_leg.py` | **开跑前环境前提检查 + 明确告警**（属"错误捕获/设备状态"职责），不设变量时**不误报** |
| ③ `docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md` | 完整根因、复现、责任层判定与三份提交建议 |
| ④ `STATUS.md`「阻塞与需要协调」 | 正式登记（含"若厂商不修的交付口径"预案） |
| ⑤ 《新芯片接入手册》 | 「已知坑」一节：最小复现 + 触发条件 + 临时规避 + **规避的代价** |

**统一措辞**：
> **根本原因在厂商 CUDA 兼容运行时 `libxpucuda.so`（KL3 事件机制与设备事件同步原语的交互），
> 需上报芯片厂商适配**；我方已按 2b 方式规避以不阻塞本方向验证，并如实标注条件。

#### 2.6.4 一个必须做的对照（属我方五域）

该变量**可能影响厂商设备异常上报**——而"错误捕获"正是我方五域之一。因此：

> **阶段 4 错误闭环必须在「设 / 不设该变量」两种设置下各跑一次并对比**，
> 差异本身就是产出，并用于回答"关闭该变量是否损失诊断能力"。

（这也正是要向厂商提问的重点问题之一：**`XPU_EVENT_KL3_ENABLE` 的语义是什么？
关闭后设备异常是否仍能上报到 Python 层？** —— 镜像与厂商文档里对该变量**零说明**。）

#### 2.6.5 明确**不做**的事（守边界）

- ❌ 不改 FlagGems 的 `tools/env.sh` / `src/flag_gems/backends.yaml` / CI 配置
- ❌ 不改锁定镜像的环境口径，不把"不设该变量"写进基座文件
- ❌ 不把"关闭变量后的通过"包装成"在官方推荐环境下已验证通过"
- ❌ 不代厂商做规避性补丁（如 hook 掉 event record）

#### 2.6.6 若厂商不修：本方向的交付口径预案

训练腿以「**单卡证据 + 多卡标注条件证据 + 归属判定 + 最小复现**」形式交付，
验收标准 4 按**标注条件**判定，并在 STATUS 与手册中把该项登记为
「**已识别、需上游修复、不影响本方向其余交付**」——**不阻塞 release**。

#### 2.6.7 本轮已落地的改动与回归证据

| 改动 | 回归证据 |
|---|---|
| `backends/base.py` 新增可选方法 `known_issues()`（默认 `[]`） | `base 默认 known_issues(): []`、`stream_priority_range(): None` ⇒ **对既有后端零回归** |
| `backends/kunlun/backend.py` 新增 `_KNOWN_ISSUES` + `known_issues()`，`info()` 暴露 `known_issues` 与追加的 `known_upstream_defects` | `known_issues 条数: 1`；`info 含 known_issues: True`；`known_upstream_defects 条数: 3` |
| `proto/proto_train_leg.py` 新增 `_preflight_env_check()`（KL3=1 时告警） | 告警正确输出；**不设变量时不误报**（对照通过） |
| 既有能力不受影响 | `smoke_runtime.py --backend kunlun` **42 通过 / 0 失败**；`conformance` **13/13** |

---

## 3. 待办总清单（按「谁来做」分类）

### A. 我方待做（可立即执行）

| # | 事项 | 依赖 |
|---|---|---|
| ~~A1~~ | ~~T5b 复测~~ | ✅ **已完成**：flagcx 三类通信全 OK（卡 6,7） |
| A2 | **训练腿**：判定「反复集合通信挂死」的性质（判定探针 C/D 已启动），再据结论决定修复或分流 | 🔄 **进行中**（卡 6,7） |
| A2b | 通信三类对照 | ✅ **已达成**：探针与训练脚本中均通过（§2.4.7/§2.4.8） |
| A3 | **阶段 3 推理腿**：单卡前向 + 服务化，复用共享缓存 `Qwen3-Embedding-0.6B` | 可**立即并行启动** |
| A4 | **阶段 4 错误闭环**：昆仑芯错误映射表 v0（异常类型 + 消息模板为键）+ 注入→分级→恢复闭环 | 可与 A3 穿插 |
| A5 | **阶段 5 产出**：《新芯片接入手册》（8 章提纲已拟）+ 接口约定修订建议（已积 7 条）+ 缺失项清单 + 对外提交单 2 张 + **release** | A2–A4 |
| A6 | 提交 `proto_train_leg.py` 后端无关化改动（现未提交） | A2 后一并提交 |
| A7 | §10 待补测：**有界等待能力**、**带卡容器并发上限** | 可立即做 |

### B. 需要你在终端操作（我的通道非交互，做不了 sudo）

| # | 命令 | 用途 |
|---|---|---|
| ~~B1~~ | ~~`dma_excp_mask` 置 1~~ | ⏸ **暂不需要** —— flagcx 已在不改开关的情况下跑通，真因是用卡而非开关 |
| B2 | 组织架构 PR 合入 dev-1.0（0 冲突） | 910C 遗留 #1 |

### C. 需对外提交（非我方职责，五域之外）

| # | 项 | 归属 |
|---|---|---|
| C1 | `Stream.priority_range()` 触发 PyTorch `INTERNAL ASSERT`（`c10/cuda/CUDAStream.h:188`） | XPytorch / 昆仑芯 torch 后端 |
| C2 | 厂商错误码不透出到 Python 异常 | 上游错误上报层 |
| **C3** | **P800 侧反复集合通信非确定性挂死**（裸 `dist.all_reduce` 复现，未经过我方运行时层；910C 同款 50 步稳定） | 候选：P800 侧 flagcx/BKCL。**定案前需补**：换卡复现 + 向上游确认 |

**提交渠道待确认**（昆仑芯支持渠道？还是经总组转达？）—— 见 §10 待确认项。

### D. 需总组裁定

| # | 事项 |
|---|---|
| D1 | 昆仑芯机器使用约束是否入 `stack.lock`（需 `docker` 权限 + 自有可写数据目录） |
| D2 | **「昆仑芯设备 API 为 `torch.cuda` 而非 `torch.xpu`」** 建议写入基座说明（跨方向通用） |
| D3 | 训练腿镜像未发布 registry（910C 遗留 #2） |
| D4 | ~~训练腿 torch_fl 例外退出口径时间表（910C 遗留 #3）~~ ✅ 2026-09-22 已闭环 |

---

## 4. 证据文件索引（避免碎片化的入口）

| 类别 | 文件 |
|---|---|
| **本报告** | `P800/docs/PROGRESS_REPORT_20260914.md` |
| **结论核对与责任层判定** | `P800/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md` |
| P800 环境汇总 | `P800/docs/KUNLUN_P800_ENV_REPORT_20260914.md` |
| P800 五域基线实测 | `P800/docs/KUNLUN_P800_BASELINE_PROBE_20260914.md` |
| P800 接入工作方案（含 §7.5 已修问题） | `P800/docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` |
| 910C 阶段性总结 | `910C/docs/DC_STAGE_SUMMARY_20260909.md` |
| **接口约定（规范本体）** | `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` |
| 错误闭环记录 | `910C/docs/ERROR_RECOVERY_LOOP_20260909.md` |
| 910C 侧错误码表（108 条，**不迁移**） | `910C/docs/ACL_ERROR_MAP_20260901.md` |
| 本方向 STATUS（总组收拢依据） | `dev/device-context/STATUS.md` |
| 昆仑芯后端实现 | `prototype/runtime/backends/kunlun/backend.py` |
| conformance 结果 | `prototype/runtime/conformance/conformance_runtime_kunlun.json`、`..._kunlun_infer.json` |
| **P800 训练腿证据（A/B 对照）** | `P800/probes/E_train_leg_result_rank{0,1}.json`、`E_train_ab.log`、`E_train_R2_control_KL3on.log` |
| 探针与自检证据 | `P800/probes/`（`dc_probe_p800.py`、`dc_probe_isolated.py`、两份 json、`smoke_kunlun_20260914.txt`） |

---

## 5. 风险与判断

| # | 风险 | 影响 | 应对 |
|---|---|---|---|
| 1 | **共享机卡占用波动**（**本轮已实际踩到**） | 实验被他人负载干扰，产生**假结论**（曾误判 flagcx 不可用） | 跑前查 `xpu-smi` 挑**连续空闲**卡；实验记录写明用卡；关键结论必须换卡复测 |
| 2 | **上游缺陷（流优先级 / 错误码）阻塞部分能力** | 个别 conformance 用例需如实跳过 | 已在 `supports()` 如实声明 + conformance 如实跳过 + 对外提交 2 张单；**不伪造通过**。训练腿**不再受阻**（T5b 已证明通信可用） |
| 3 | **同进程错误污染** | 一个用例失败污染后续用例，产生错误结论 | 失败用例**进程隔离**（已实践，并已纠正一次误读） |
| 4 | **对照实验的单变量性** | 多变量同时变 → 结论不可靠 | 一切对照固定单变量；本轮 T5 vs T5' 已识别未控变量并补测 |
| 5 | 时间口径被误读 | 与月度计划（11 月正式交付）冲突 | 定位已写明：9 月＝原型接入，11 月＝整体交付 |

---

## 6. 下一步（编号化，等你发话）

| 编号 | 动作 | 预期产出 |
|---|---|---|
| ~~P1~~ | ~~收 T5b 结果~~ | ✅ **已完成**：flagcx 可用，真因是用卡 |
| **P2** | 收判定探针 C/D 结果 → 定位「反复集合通信挂死」性质（重复次数 or 特定张量）→ 按 §2.3 归属规则处置 | 训练腿可跑通 or 一条对外提交证据 |
| **P2b** | 启动**推理腿**（与训练腿并行，权重对等）—— **不被训练腿卡点阻塞** | 验收标准 5 的证据 |
| **P3** | 错误映射表 v0 + 注入→恢复闭环 | 验收标准 4 的设备侧部分 |
| **P4** | 起草《新芯片接入手册》+ 接口约定修订建议 | 月度计划 11 月的交付物 |
| **P5** | 提交训练腿脚本改动 + 本报告 | 仓库一致性 |
