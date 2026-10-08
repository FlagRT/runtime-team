# P800（昆仑芯）· 第二个芯片接入实例（分支看板）

> 定位：**统一运行时原型的第二个接入实例** —— 按接入规范**新建** `kunlun` backend，
> 而不是「把 910C 的能力适配/移植过来」。
> 迁移的是**规范与方法**；910C 的实现与结论**不迁移**（见 `../910C/README.md` §1 铁律）。
> 状态：✅ **阶段 0–5 全部完成**（接入 → 训练腿 → 推理腿 → 错误闭环 → **官方 `-base` 镜像等价性验证** → **阶段 5 收敛三件套**）
> 上级看板：`../README.md` ｜ 通用规范与原型：`../prototype/` ｜ 910C 实例：`../910C/`

---

## 0. 结论速览

> 下表**按时间倒序**（越上越新）；历史批次保留不删，标注为「历史批次」。
| 项 | 状态 | 关键数字 |
|---|---|---|
| ⭐ **D2 调度效果对照实验（10-08 · 第十三轮）· 最新** | ✅ **如实 `NOT_APPLICABLE`**（区间 `(0,0)` **唯一档** ⇒ 无从对照）· 仪器有效性已验（正对照 **8/8 / +9.87 ms**）· ⭐ 已通过**「不适用自证审计」**（逐卡 **7/8** · driver(`cu*`)+runtime(`cuda*`) **两层一致** · 全量 `nm -D` 扫描） | `../prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54** · `../prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L63–L78** |
| ⭐ **台账 E1：职责审计扩到 78 项（10-08）** | ✅ `DUTY_RESPONSE_PASS` **67 / 0 / 11**（SKIP 均如实不具备）· 非空转 **31 / 8 / 0** · 离线 **106/0/1** | `../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92** |
| ⭐ **(A) 真落地后的全套复跑（09-30 r8）** | ✅ 全绿无回归：离线 **106/0/1** · 释放探针 **10/10**（该流**真能承载算子**）· 训练腿 6/6 · 推理腿 13/13 | `../prototype/docs/STREAM_PRIORITY_API_20260930.md` **L265–L348**（§7 v2 更新段） |
| 流优先级统一 API 落地（09-30 r7） | ✅ **`STREAM_PRIORITY_API_PASS` 7/7**；范围查询走**真原语** ⇒ `(0,0)` **退化单点**（旧版返回 `None` 的「根因表述」已更正） | `../prototype/docs/STREAM_PRIORITY_API_20260930.md` **L11–L28** · **L273–L301**（§7.2 为何可以声明） |
| 工作包 B/C 接口落地（09-29） | ✅ B **3/3** · C 改为**只读观测** `context_query`（真机 `managed_by="external"`、`readonly_safe=true`） | `../prototype/docs/WORKPACKAGE_BC_INTERFACE_20260929.md` **L24–L40 / L252–L297** · `docs/KUNLUN_CONTEXT_SEMANTICS_20260929.md` |
| 跨实例复验 r2（09-29） | ✅ 10 项全绿 · 0 回归；⚠️ 首轮在**卡 1** 挂死经**单变量对照**判为**卡级环境问题**（非本层回归） | `docs/KUNLUN_P800_REGRESS_AFTER_FIX_20260929.md` |
| 职责响应审计（09-28 · 39 项口径） | ✅ `DUTY_RESPONSE_PASS` **36 / 0 / 3**；首轮暴露 `recover_device` **缺 `state`** ⇒ 已补做 | `docs/DUTY_RESPONSE_AUDIT_P800_20260928.md` |
| 三芯片职责验收（09-22 傍晚 · 历史批次） | ✅ 10 项全绿（离线 39/0 · 冒烟 46/0 · conformance 13+6 · 训练腿 6/6 · 推理腿 13/13 · 服务化 PASS · 错误闭环 5/0/0） | `../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` |
| 阶段 0–5 全部完成（09-14 → 09-20） | ✅ 环境与五域基线 → 单卡接入 → 训练腿（loss 15.4488→11.1481、3482→**3533.5 tok/s**）→ 推理腿前向 **13/13**（53.12 句/s）→ 服务化 **10/10**（30.70 句/s）→ 错误闭环 **5/0/0** → 收敛三件套 | `docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` · `docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` · `docs/PROGRESS_REPORT_20260914.md` |
| 官方 `-base` 镜像等价性（09-20） | ✅ 全部结论在官方推荐镜像上复现（conformance 逐用例一致 · 推理腿 `detail` **14/14 逐字相同** · **KL3 挂死一致重现 A 3/3**）⇒ 建议以官方 `-base` 入锁 | `docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` |
| 多流 Stream 16 项基线（09-20） | ✅ **14 通过 / 1 如实不支持（S-12 流优先级，上游缺陷）/ 1 不适用**；探针 **8/8**（与 910C 逐项一致）· S-7 图捕获 **5/5** · S-16 **2000 流** | `docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md` |
| 已知厂商缺陷（2 条 · 均归属厂商侧） | ⚠️ ① KL3 事件同步概率性挂死（≈89%，**不影响单进程设备上下文路径**）② **物理卡 1 计算通路故障**（控制面正常 ⇒「能查到卡」≠「能用卡」） | 本文件 **§3.1 / §3.2** · `docs/KUNLUN_P800_CARD1_HANG_ISSUE_20260923.md` |
| 框架缺陷（第 4 例） | ✅ 已修在框架层：错误对象**跨模块类不相等** → `disposition` 取 `KeyError` | `docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` **§3** |

> 全量文档的效力分层与一句话说明见主看板 §6.4.4；复核入口见 `../prototype/docs/VERIFICATION_MANIFEST_20260920.md`。

---

## 0.5 ⚠️ 环境事实与环境口径冲突（2026-09-30 实测，避免后来者白找）

| 事实 | 内容 |
|---|---|
| **本机没有宿主 git 副本** | `/workspace/runtime-team`、`/data2/hliu553/runtime-team` **都不存在**。P800 侧只有**同步过去的 `prototype/` 目录**（宿主 `/data2/hliu553/dc_regress_20260929/prototype` = 容器内 `/workspace/...`）。**不要按"应该有一份"去找**；需要版本信息请看 910C 宿主副本（已对齐）或 GitHub。 |
| **路径映射** | 宿主 `/data2/hliu553` = 容器 `/workspace`。**在宿主上执行的脚本**其重定向要用宿主路径，**传给容器程序的 `--out`** 要用容器路径 —— 混用会得到一堆 rc=1（2026-09-30 首跑即如此）。 |
| **流优先级：可设置，但只有一个档位**（2026-09-30 v2 修订） | ⭐ 本层**声明** `stream_priority_control`：取值域 = 设备自报区间 `(0, 0)`（单点）⇒ 「请求 0 = 回读 0」成立；实现走 `cuStreamCreateWithPriority` + `torch.cuda.ExternalStream` 包装（真机实测该流**真能承载算子**）。⚠️ **单档 ⇒ 设置不产生调度区分**（`known_issues: KUNLUN-STREAM-PRIORITY-SINGLE-LEVEL`）。**能设置 ≠ 有效果**，别据此做调度决策。
| **旧口径（v1，已被上一条取代）** | 兼容层三个入口**都在**（`cuCtxGetStreamPriorityRange` / `cuStreamCreateWithPriority` / `cuStreamGetPriority`），但范围查询返回 **`least=0, greatest=0`** ⇒ **本机没有可调的优先级空间**；且 `cuStreamCreateWithPriority(..., prio=-1)` **回读仍为 0**（不保留传入值）⇒ 优先级在 P800 上**无可观测效果**（不是「接口没实现」，而是「设备只报一个档位」）。证据 `probes/prio_readback_kunlun_20260930.log`；重跑 `probes/inspect_prio_readback_kunlun_20260930.py` |
| **⚠️ 一处口径冲突（2026-09-30 新观察到）** | 厂商栈在 stderr 提示「官方手册要求测试前 `export XPU_EVENT_KL3_ENABLE=1`，否则**部分路径行为未定义**」—— 而**该变量正是本方向 09-14 实测的挂死触发条件**（KL3 事件 + 设备集合通信 ⇒ `cudaDeviceSynchronize` 永久自旋，见 `docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md`）。⇒ **手册要求设置的环境变量 = 我们实测的故障触发条件**。本方向所有取数（含 09-30 新探针）**均在未设该变量的状态下**完成；与既有「错误闭环两设置逐字节一致」的记录相容，但厂商那句声明如实标注为**上游口径冲突**，已并入上报材料 |
| **选卡** | `dev1` = **已知故障卡**（UUID `b3509946…`，`0 MiB / 0%` **与好卡同貌**，靠 UUID 识别）⇒ **永远避开**；2026-09-30 复跑用 **`dev4`**（292 MiB / 0%）。 |

---

## 1. 本目录放什么

| | 内容 |
|---|---|
| ✅ **放** | P800 **这一实例**的验证资产：环境汇总、五域基线实测、接入工作方案、根因核对报告、全量进度报告、探针脚本与**原始证据** |
| ❌ **不放** | `kunlun` backend 代码（属通用原型，在 `../prototype/runtime/backends/kunlun/`）；conformance 用例与结果（在 `../prototype/runtime/conformance/`） |

```
P800/
├── docs/                            # 见 §4
│   ├── KUNLUN_P800_ENV_REPORT_20260914.md
│   ├── KUNLUN_P800_BASELINE_PROBE_20260914.md
│   ├── KUNLUN_P800_ADAPT_PLAN_20260914.md
│   ├── KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md
│   └── PROGRESS_REPORT_20260914.md
└── probes/                          # 探针脚本 + 原始证据（见 §5）
```

---

## 2. 关键认知（写进《新芯片接入手册》，均经实测）

| # | 认知 | 证据 |
|---|---|---|
| **1** | **设备 API 走 `torch.cuda`，`torch.xpu` 不可用** | `torch.xpu.is_available() = False`（AssertionError: Torch not compiled with XPU enabled）；`torch.cuda.device_count() = 8`；编译标志 **`USE_XPU=OFF`**；官方 xpu3.6 单测 `conftest.py` 的 `--device` 默认值即 `'cuda'`。机制为 XPytorch + `torch_xray` 符号重写 |
| **2** | **选卡变量是 `CUDA_VISIBLE_DEVICES`** | 实测 `=2` → `device_count()=1`；`=2,5` → `2`。它才是 910C `ASCEND_RT_VISIBLE_DEVICES` 的对应物，**不是** XPU 侧变量 |
| **3** | **同一 FlagCX，两芯片后端名不同** | 910C 为 **`hccl`**（路线 B 时期曾注册为 `flagos`）；P800 注册为 **`flagcx`**，且**必须显式 `import flagcx`** 才会注册；用法 `init_process_group("cpu:gloo,cuda:flagcx")` + `FLAGCX_ADAPTOR=klx` |
| **4** | **只有 `flagcx` 这一条通信路径可用** | `nccl` 挂死；`xccl` 未编译（`Distributed package doesn't have XCCL built in`）；`kccl` 无响应 |
| **5** | **HF cache 要指向 `snapshots/<hash>`** | 传仓库根目录报 `Unrecognized model ... Should have a model_type key`（根目录只有 `blobs/`、`refs/`、`snapshots/`） |
| **6** | **镜像本机已有，无需联网** | `flagtree-xpu3.6-...-flaggems-main-dev:202608`（38.3 GB）已在本地镜像库 → 官方手册的 59.9 GB `pull` 与 32 GB `load` 全部跳过；FlagGems 源码亦已在容器内 `/env/FlagGems` |

**环境差异（相对 910C 的部署前置，建议入基座说明）**

| 项 | P800 现状 |
|---|---|
| 连接 | 非标准端口 **26008**（`~/.ssh/config` 的 `Host P800` 曾缺 `Port` 行） |
| 权限 | 需加入 `docker` 组 + 需自有可写数据目录（`/data2/hliu553`） |
| 镜像落盘 | `/var/lib/docker` 已 **bind mount 到 `/data1`**（5.8 TB NVMe），充足 |
| 共享程度 | 25 人在线、22 个容器；**用卡前必须 `xpu-smi` 挑「连续且空闲」的卡并记录用卡**，且**挑完先跑最小计算/事件探针**再正式取数；⚠️ **smi 索引 1（物理卡 1）为已知故障卡，禁止用于计算类验证**（见 §3.1） |
| 拓扑 | XPU0-3 属 NUMA0、XPU4-7 属 NUMA1；组内 XL 私有链路、跨组 SYS；NIC 与卡 PIX 直连 |

---

## 3. 已知厂商缺陷（⚠️ 需芯片厂商适配，本方向不阻塞）

### 3.1 已知厂商缺陷：`XPU_EVENT_KL3_ENABLE=1` 下设备事件同步原语概率性永久自旋

**现象**：`XPU_EVENT_KL3_ENABLE=1` 且存在设备侧集合通信时，设备事件同步原语**概率性永久自旋**。

| 项 | 内容 |
|---|---|
| 项 | 内容 |
|---|---|
| **判别条件** | **两要素**，缺一不挂：① `XPU_EVENT_KL3_ENABLE=1`；② 存在设备侧集合通信 |
| **复现率** | **18 次运行 16 次挂死（≈89%）**；挂死步数游走（rep 0/20/30/40/70/100）；与数据量 / 形状 / reduce op / 用哪对卡 / 同步间隔**均无关** |
| **责任层** | **厂商运行时/驱动层**（`libxpucuda.so.515.58.kunlun`）；算子层（`flag_gems` 0 引用）与编译层（无编译产物变更）**均已硬证据排除** |
| **规避 + 代价** | **不设** `XPU_EVENT_KL3_ENABLE`（我方两腿均不依赖 FlagGems）；⚠️ 但它是 FlagGems kunlunxin 后端的**官方推荐变量** ⇒ 关闭是否损失异常上报**须上游确认**，本方向不擅改公共资产 |
| **上游旁证** | FlagOS 官方 `build-infra/configs.yaml` 原文：*"XPU_EVENT_KL3_ENABLE deliberately NOT set: it is the P1 fake-hang trigger …"* ⇒ **官方明确不设** |
| **机器可读** | `kunlun` 后端 `info()["known_issues"]`（12 字段结构化：复现率 / 责任层 / 规避 / 上报对象）+ `proto_train_leg.py` 开跑前告警 |

> **统一措辞**：根本原因在厂商 CUDA 兼容运行时 `libxpucuda.so`（KL3 事件机制与设备事件同步原语的交互），
> **需上报芯片厂商适配**；我方已按上述方式规避以不阻塞本方向验证，并如实标注条件。
>
> **⚠️ 补充（2026-09-22）：P800 存在两条镜像血统，方向侧不自行切换** ——
> ① 现用 **FlagTree 线** `flagtree-xpu3.6-…:202608-base`（= FlagTree 手册给 P800 的唯一镜像，
> 与类脑 x-benchmark 指向同一条 xpu3.6 线，**镜像本身无需调整**）；
> ② **FlagOS 官方线** `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-kunlunxin-xre5.37.1:2.2.0`
> （flagtree 0.7.0rc2+xpu3.6，但**底层 SDK 换代到 XRE 5.37.1**，前置要求宿主驱动 **5.37.1**，
> 而我们实测宿主为 **5.0.21.47**）。
> **建议总组明确走哪条**；完整对照见 `../prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md`。

### 3.2 已知厂商侧缺陷（**卡级 · 单卡硬件**）：物理卡 1 设备执行通路故障（2026-09-29 登记）

**结论**：**物理卡 1**（smi 索引 1 = `/dev/xpu2` = UUID `GPU-b3509946-0bc6-5744-bd7c-a0b87dadef02`
= SN `02K15K0263D002KN` = PCI `0000:16:00.0`）上**任何设备计算内核都无法完成执行**
⇒ **禁止用于任何计算类验证**。**控制面查询仍正常，所以"能查到卡"不等于"能用卡"。**

| 项 | 内容 |
|---|---|
| **症状** | 计算内核不完成：`torch.cuda.synchronize()` **永不返回**（信号亦不可中断，只能外层 `timeout` 强杀）。我方另观测到**更靠前的一层**：`Event.record + wait_host` 即失败、`probe_device` 卡死 |
| **控制面 vs 计算面** | **控制面正常**（`get_device_properties` / `mem_get_info` 均正常返回 `(103045660672, 103079215104)`），**只有计算内核不完成** ⇒ 极易被误判成"程序逻辑问题" |
| **内核日志** | `KL_XID_KERNEL_EXCEPTION` + `kl3_wait_for_noc_idle() timeout` + `cluster[N]: ..reason[29] task timeout`；驱动直接打印出错内核符号（最小复现中是 `xpukernel_xpu3::constant<float>`，即 `torch.ones(4)` 的 fill 内核） |
| **不是我方负载引入** | 该卡同类异常**最早 2026-09-17**（**早于我方首次使用该卡**）；同类异常 20 次中 **12 次来自其他用户互不相关的作业**（含一个 vLLM 推理引擎），失败算子涵盖 fill / 随机数 / H2D / 转置 / `get_cluster_clock` / AllReduce ⇒ **执行通路层面故障，不是某应用或算子的 bug** |
| **"看占用"挑不出来** | 挂死时该卡 `Memory_used = 0 MiB`、`util = 0%`、`fuser` 无句柄持有者 —— 在 `xpu-smi` 上与别的空闲卡**完全一样** |
| **责任层** | **厂商侧**（卡硬件 / 驱动执行通路）；已由对外正式问题报告提交 ⇒ **本方向不阻塞** |
| **规避** | 用卡时**避开 smi 索引 1**；我方 09-29 回归改用卡 4 后 **10 项全绿**。选卡后**先跑最小计算/事件探针**再取数 |
| **原始报告（归档）** | ⭐ [`docs/KUNLUN_P800_CARD1_HANG_ISSUE_20260923.md`](docs/KUNLUN_P800_CARD1_HANG_ISSUE_20260923.md)（对外报告**逐字归档** + 我方交叉核对；含最小复现、内核原文、七类已排除假设） |
| **我方判别证据** | `probes/DIAG_kunlun_card1_event_hang_20260929.log`（A/B/C 三组 + **卡身份硬核对**） |
| **历史结论复核** | ✅ 已审计：P800 全部**在册结论**取自卡 **4 / 5 / 6（6,7）**；唯一一次触碰卡 1 的 09-14 记录本就已撤销为「无效证据」⇒ **无需复核任何已发布结论** |

> ⭐ **给后续接入者的两条纪律**（已写进本目录「环境差异」与 skill 手册的用卡章节）：
> ① **`xpu-smi` 显示空闲 ≠ 该卡功能正常** ⇒ **先跑最小计算/事件探针**再取数；
> ② 判定"是不是我们改坏的"用**单变量对照**：把**上一版原型**放到**同一张卡**跑**同一条命令** ——
> 同样挂死即环境问题（我方本轮据此在 30 分钟内定案，**未误改任何判据**）。

---

## 4. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/KUNLUN_P800_ENV_REPORT_20260914.md` | **任务 1 环境汇总**：主机/系统、8× P800、内存/CPU、数据盘与权限、网络与拓扑、可复用资产 |
| `docs/KUNLUN_P800_BASELINE_PROBE_20260914.md` | **五域基线实测**：设备抽象 / 多流 Stream-Event / 算子 / 错误 / 分布式，含与 910C 对照与 3 条缺失项 |
| `docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` | **接入工作方案**：定位与交付边界、接入路线（§2.2 后端划分依据）、验证与验收标准（6 条）、阶段计划、风控、对外提交物、**§7.5 接入过程暴露并已修的 3 个框架缺陷** |
| `docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md` | **结论核对与责任层判定**：准确性 / 可复现性 / 该提算子层还是编译层（三问全答） |
| `docs/PROGRESS_REPORT_20260914.md` | **全量进度报告**（910C 回顾 + P800 主体 + 待办总清单按「谁来做」四分类 + 证据索引 + 风险与下一步） |
| `docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` | **阶段 3/4 验证报告**：推理腿 13/13 与 910C 同构对照、错误闭环两设置对照（逐字节一致）、**§3 第 4 个框架缺陷的根因与修复**、待办 |
| ⭐ `docs/DUTY_RESPONSE_AUDIT_P800_20260928.md` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **36 OK / 0 FAIL / 3 SKIP**（SKIP 均如实不具备）；**首轮即暴露 `recover_device` 缺 `state`** ⇒ 补做后 E3 转 PASS |
| ⭐ `docs/KUNLUN_P800_REGRESS_AFTER_FIX_20260929.md` | **三处层内修复后的全套回归 + 一处卡级环境问题的判别**：10 项全绿 / 0 回归；卡 1 挂死的 A/B/C 三组判别（**判定 = 环境问题，非代码缺陷**）；新增「**空闲 ≠ 正常**」选卡纪律；命名口径（无后缀 = r1，`_r2` = 本轮） |
| ⚠️ `docs/KUNLUN_P800_CARD1_HANG_ISSUE_20260923.md` | **对外正式问题报告的逐字归档**（物理卡 1 同步挂死）：最小复现（与框架无关，3 行 PyTorch）、内核原文与出错内核符号、**七类已排除假设**；附我方**卡身份硬核对**（Serial/UUID/Minor/PCI 逐项一致）。**不迁移**（卡级结论） |
| `docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` | **官方 `-base` 镜像等价性验证报告**：**§0 镜像速查**（两镜像 tag/digest/大小/来源一把看全 + 官方镜像获取与补齐三步 + 容器启动参数对照）· 全部结论复现对照（逐用例/逐 `detail`）· **KL3 挂死一致重现** · **`-base` 开箱缺 `triton` 的调用链与补齐命令** · 对镜像入锁的建议 |
| `docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md` | **多流 Stream 验收基线 16 项逐项比对报告**：16 项 P800 结论 + **与 910C 逐项对照**（仅 S-12 一项差异）+ **S-7 图捕获首测 5/5** 与**一处自我纠错（首测失败实为用法错误）** + 证据形态差异说明 + 复现命令 |
| `../prototype/docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md` | **跨实例参考**（910C + P800 并列）：镜像 / 模型 / 训推框架 / 参数逐项对照 + **依据链** + 复用坑清单；后续接入者与框架方向首读 |

**规范与原型（在 `../prototype/`，不属本目录）**

- 接入规范本体：`../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` §2「Backend 插件接入规范」
- 事件语义契约：`../prototype/docs/event_semantics_contract.md`
- `kunlun` backend 实现：`../prototype/runtime/backends/kunlun/backend.py`
- conformance 结果：`../prototype/runtime/conformance/conformance_runtime_kunlun.json`（13/13）、`..._kunlun_infer.json`（6/6）

---

## 5. 证据索引

**证据索引（`probes/`）** —— 探针脚本清单、逐份日志与 JSON 的说明已移入
[`docs/EVIDENCE_INDEX_P800.md`](docs/EVIDENCE_INDEX_P800.md)；本看板只留入口：

| 用途 | 证据入口 |
|---|---|
| 当前结论 = 哪一份 | `probes/audit_20261008_out/`（自证审计）· `probes/d2_20261008_out/` · `probes/e1_regress_kunlun_20261008_out/` · `probes/r8_regress_kunlun_20260930*` |
| KL3 缺陷取证 | `probes/A_round1_battery_20260914.log` · `B_round2_repeat_*` · `C_round3_dose_*` · `D_verify_truthvalue_*` · `E_train_ab.log` |
| 卡 1 故障判别 | `probes/DIAG_kunlun_card1_event_hang_20260929.log`（A/B/C 三组 + 卡身份硬核对） |
| 两条腿 / 服务化 / 错误闭环 | `probes/E_train_leg_result_rank{0,1}.json` · `F_infer_leg_result_20260920.json` · `F2_serve_result_20260920.json` · `error_recovery_loop_kunlun_KL3{on,off}.json` |
| 口径与命名规范 | `../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5 |

---

## 6. 下一步

**执行前置（把最新原型与脚本送进容器）**

```bash
tar czf /tmp/dc.tgz -C dev/device-context prototype P800/probes \
    --exclude='__pycache__' --exclude='*.tgz'
scp /tmp/dc.tgz P800:/data2/hliu553/            # 端口 26008
ssh P800 'cd /data2/hliu553 && tar xzf dc.tgz --overwrite && rm dc.tgz'
```

**任务表**

| # | 动作 | 状态 | 结果 / 判据 |
|---|---|---|---|
| 1 | **阶段 3 推理腿（单卡前向）** | ✅ **09-20 完成** | `INFER_LEG_PASS 13/13`（1 项如实跳过）：维度 **1024** ｜ 区分度 **0.6392** ｜ **53.12 句/s** ｜ p50 **56.17 ms** |
| 1b | 阶段 3 推理腿（vLLM 服务化形态，`--runner pooling --convert embed`） | ✅ **09-20 完成** | `SERVE_LEG_PASS 10/10`：区分度 **0.4102** ｜ **30.70 句/s** ｜ p50 96.4 ms ｜ 超长输入 → **L2_PARAM/raise** + 业务继续。⚠️ 硬前置 `PYTHONPATH=/env/FlagGems/src`（详见验证报告 §1.3） |
| 2 | **阶段 4 错误闭环（两设置对照）** | ✅ **09-20 完成** | 两组均 `ERROR_RECOVERY_LOOP_PASS`（闭环 **5 / 跳过 0 / 失败 0**），**逐字节一致** ⇒ 关闭 KL3 **不损失**诊断能力 |
| 2b | **官方 `-base` 镜像等价性验证** | ✅ **09-20 完成** | 全部结论复现：conformance 13+6 逐用例一致、smoke 42/0、训练腿/推理腿/服务化全 PASS、**KL3 A 组 3/3 挂死 + B 组 2/2 通过** ⇒ 缺陷与镜像无关。⚠️ `-base` 开箱**无 `triton`**，须按官方手册装 `flagtree===0.7.0rc3+xpu3.6`（3.3 GB） |
| 3 | **阶段 5 收敛**：《新芯片接入手册》（8 步流程 + 验收清单 13 项，§2 六条认知与 §3 已知缺陷已并入）；接口约定修订建议 **6 条**；原型 release **`runtime-v0.2.0`** | ✅ **09-20 完成** | 手册 `../prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`；修订建议 `../prototype/docs/INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`；release `../prototype/RELEASE_NOTES_v0.2.0.md` |
| 4 | **上报渠道待确认**：直连昆仑芯支持，还是经总组转达 | ⏳ 待定 | STATUS「阻塞与需要协调」已登记 |

**本次为执行做的准备（2026-09-20）**

- `../prototype/runtime/proto/proto_infer_leg.py` → **后端无关化 V2**：路径/模型走 `DC_*` 环境变量、
  设备串取 `runtime.current().device_type`、同步走 `runtime.synchronize()`、真实异常注入替代伪造错误码、
  补 p50/p90 时延；`DC_MODEL` 支持 hub 的 `models--xxx` 目录（自动解析 snapshot）。
- `../prototype/runtime/proto/proto_error_recovery_loop.py` → 补 `DC_BACKEND` 环境变量（与另两条腿一致）。
- 新增 `probes/F_infer_leg.sh`、`probes/G_error_loop.sh`（超时兜底 + 挂死快照 + 用卡复查，沿用 battery 体例）。

**环境核对结论（2026-09-20 只读实测）**：容器 `hliu553-device-context-p800` Up 5 天；
conda env `python310_torch29_cuda`（py3.10.18 / torch 2.9.0+cu129 / transformers 4.57.1 /
sentence_transformers 5.7.0 / **vLLM 0.13.0** / flagcx 可导入）；`torch.cuda.device_count() = 8`；
共享缓存模型 `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614…`（挂载 `/hf_cache`）。
