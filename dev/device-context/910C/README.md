# 910C（昇腾）· 第一个芯片落地实例（分支看板）

> 定位：**统一运行时原型的第一个落地实例** —— 设备上下文（device-context）与多流 Stream 在**昇腾 910C** 上的完整验证。
> 状态：✅ **已完成**（训练腿与推理腿均闭环，错误注入→恢复闭环通过）
> 上级看板：`../README.md` ｜ **通用规范与原型：`../prototype/`（芯片无关，不在此目录）**

---

## 1. 本目录放什么 / 不放什么

| | 内容 |
|---|---|
| ✅ **放** | 910C **专属**的落地实例资产：分布式训练与推理的既有工作、昇腾专属文档与结论（ACL 错误码表、910C 实测映射、CANN/镜像约束、阶段总结） |
| ❌ **不放** | **统一运行时原型与通用规范** —— 它们在 `../prototype/`（`runtime/api`、13 个抽象、conformance 用例、接口约定、事件语义契约）。这些是**芯片无关**的，新芯片按同一份规范接入 |

> ⚠️ **铁律（新芯片方向务必先读）**：本目录里的**实现与结论不迁移**。
> 910C 是规范的**第一个**落地实例，不是规范的载体。新芯片（如 P800）应
> **按 `../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` §2 的 Backend 插件接入规范新建实例**，
> 迁移的是**规范与方法**；本目录的 `backends/ascend|flagos` 绑定、**108 条 ACL 错误码表**、
> CANN 约束、镜像基座、并发上限 3、`ASCEND_RT_VISIBLE_DEVICES` **一律不迁移**。

---

## 2. 目录结构

```
910C/
├── distributed_training/            # 分布式训练既有工作
│   ├── ascend_regression/           #   训练侧 conformance、探针、通信补丁验证、原始结果
│   ├── docs/                        #   训练映射、flagcx 缺陷修复、双缓冲/netcc 调查、4090 报告
│   ├── patches/                     #   FlagCX 缺陷补丁（O2/O3/O4/P2/P6/P7/P9）
│   └── scripts/                     #   训练脚本、通信工具、环境搭建
├── distributed_inference/           # 分布式推理既有工作
│   ├── inference/                   #   推理探针、错误码工具、TP 对照、服务化脚本与结果
│   ├── docs/                        #   推理映射、P0–P3 阶段报告、Qwen3 TP 对照
│   └── start_infer_container.sh
├── probes/                          # 910C 侧验证证据（跨后端脚本/标准脚本的原始日志）
└── docs/                            # 910C 专属文档（见 §4）
```

---

## 3. 关键成果（实跑证据）

> 下表**按时间倒序**（越上越新）；历史批次保留不删，标注为「历史批次」。
| 项 | 结果 |
|---|---|---|
| ⭐ **第 10 轮复跑（r10 · 10-08）· 最新** | ✅ **20 项全绿**（含两条腿 + 服务化四形态） | 台账 **D4 收尾**：`ea503cc` 的共享层改动（`create_stream` 优先级门禁 + **新增 `release_stream` / `owns_stream`** + `check_stream_usable` 收紧「已释放」）在 910C **补齐覆盖** —— 离线 **90/0/1**（r9 为 88，+2 来自新增 `[8b-②]` 段）· 对称性 **7/0** · 冒烟 **52/0** · conformance 13+6 · 契约不变式 4/4（I1④ **16/17**）· 职责审计 **39/0/0** · 错误闭环 5/0/0 · B/C 探针 PASS · 优先级 API **8/8**。⭐ **新增 4 项覆盖**：公开入口 **`ENTRY_VERIFY_PASS`**（`handle_error` 端到端）· **流所有权/释放 `STREAM_RELEASE_CONTROL_PASS` 6/0/2**（默认路径的流**不被登记**、`release_stream` 如实 **no-op 返回 False**；区间 `(0,7)` 非单点 ⇒ R2 跳过，未声明 `control` ⇒ R3 私有原语**显式拒绝**而非静默放行）· 根解析自检 **38/0** · 多卡多进程 `real` 压测 **`MULTIPROC_REAL_RECOVER_PASS`**（3 rank × 30 轮 · 11.0 s）。两条腿 **`TRAIN_LEG_PASS 6/6`**（loss **15.4498→11.1479** · 4196.6 tok/s）· **`INFER_LEG_PASS 14/14`**（dim 1024 · 68.82 句/s · 区分度 **0.6391**，与 r9 逐项相同）· 服务化 **4 形态全 `SERVE_STANDARD_PASS`**（TP=2 实测 `world_size=2`+`Worker_TP0/TP1`+`backend=hccl`；EAGER=0 实测 `enforce_eager=False`+`CUDAGraphMode.PIECEWISE`）+ `SERVE_LEG_PASS` **10/10**。⚠️ **过程遇到 1 个机器级阻塞（与本层无关）**：共享 `models/Qwen3-Embedding-0.6B/model.safetensors` 被**存储层静默损坏**（2 张量 **34396** 个非有限值；**同机 Qwen2.5-1.5B / Qwen3-4B 两款全干净**；所在 `md127` RAID5 **降级 `[4/3] [UUU_]`**）⇒ 两条腿一开始 `loss=nan` / 区分度 nan；**逐卡对照 3 张卡全 NaN、CPU 前向同样 NaN、裸字节 `ff ff`** ⇒ 判为非本层问题；改用 P800 位级原件（**sha256 == HF blob 名 `0437e45c…`**，内容寻址自证）在 scratch 目录自证修复（**只搬 2.33 MiB / 全量 1.19 GB**），修复后两条腿跑通。**共享资产未改动**。见 `docs/ASCEND_910C_R10_RERUN_20261008.md` |
| ⭐ **流优先级统一 API 落地（(A) 方案 · 09-30 第九轮）** | ✅ **`STREAM_PRIORITY_API_PASS` 8/8 · 破坏面 r9 全绿** | 用户裁定「补能力」后落地 `create_stream(priority=None)` + `stream_priority_readback()`；能力键**拆三把钥匙**（`stream_priority` 读范围 / `stream_priority_control` 能设置 / `stream_priority_readback` 能回读）—— 同 `device_state` / `device_state_control` 的拆法：**能读 ≠ 能改**。**四道约束**（基类唯一实现）：未声明 control ⇒ `NotImplementedError`（**显式拒绝，不静默降级**）· 非 int / 越界 ⇒ `ValueError` · **创建后强制回读校验**（回读 ≠ 请求 ⇒ `RuntimeError`）· `None` 行为逐位不变。**本家如实结论 = 只读**：范围可读 `(7, 0)`、回读可用（`aclrtStreamGetPriority`），**但不声明设置** —— 6 条证据钉死受阻点：torch_npu **无 `ExternalStream`**（同文件里有 `ExternalEvent`）· `Stream(stream_ptr=…)` **被接受却静默忽略** · `libtorch_npu.so` 的 `getStreamFromExternal` **未暴露到 Python** · `acl_rt.h` 无 setter；而 pyACL 本身**接受并保留**该参数（回读 7）⇒ 障碍在「**把裸 ACL 流包回 torch**」这一步，不在 ACL 层（新探针 D6 现场对照）。⚠️ **同时修掉一处契约违约**：`stream_priority_range()` 原**直接透传 pyACL 三元组** `(7,0,rc)`，而 cambricon 返回 2 元组 ⇒ 同一份下游代码跨实例读到**不同形状**（台账第 25 条）。判据：离线新增 `[8b]` 段（5 条）+ **5 处注入的非空转验证（5/5 当场 FAIL）**。**第 9 轮回归**：离线 **88/0/1** · 对称性 **7/0** · 冒烟 **52/0** · conformance 13+6 · 契约不变式 4/4（I1④ **16/17**）· 职责审计 **39/0/0** · 错误闭环 5/0/0 · B/C 探针 PASS · 流语义 **8/8** · 配额 **3/3** · 训练腿 **6/6**（4163.0 tok/s）· 推理腿 **14/14** · 服务化 **`SERVE_STANDARD_PASS`**（就绪 45 s）。见 `../prototype/docs/STREAM_PRIORITY_API_20260930.md` |
| ⭐ **B2 收尾（v2）：优先级 + 配额**（09-30 工作包 D）**｜已按官方文档核查根因并更正 v1 两处结论** | ✅ **真实配额上限 = 1979 条** · ⚠️ **优先级两层障碍** | **配额（v1 取数无效，已更正）**：`acl.rt.get_stream_available_num()` = **1979**，pyACL 直连**恰好建满 1979 条**、第 1980 条失败 `rc=207005`（`ACL_ERROR_RT_RESOURCE_ALLOC_FAIL`）⇒ **真实上限 1979**；与官方「Atlas 训练系列 Stream 最大数 2048，N = 2048 − 默认 − 内部同步」对得上（差 69）。⚠️ v1 用 `torch.npu.Stream()` 建到 100 万条 —— 实测它**构造时不消耗设备配额**（available_num 建 1000 条不变）⇒ v1 测的是 Python 对象数，不是设备上限。**优先级（结论成立，根因更硬）**：① 用 C API **回读**（`aclrtStreamGetPriority`，pyACL 未暴露、ctypes 调真实库）：pyACL `create_stream_with_config` **0/3/7 全部保留** ✅，而 `torch.npu.Stream(priority=7)` **回读恒 0** ⇒ **插件层就把参数丢了**；② 官方文档：**Atlas 训练系列上 `priority` 是「预留参数、暂不使用」**（仅推理系列真正支持 [0,7]，0 最高）；③ 新探针按机制重测（**两流同时排队**）仍是 **0/8**。⚠️ 更正：v1 把 `range[1]=7` 当高优先级，**方向标反**；P800 回读显示 `least=0,greatest=0`（**优先级空间退化为单点**）。见 `prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md`（v2） |
**优先级效果**：范围可读 `(7,0,rc=0)`、两端都能建流且算得对（各 89440）；但**实测未观测到效果** —— 低先提交时高优先级 **0/8** 轮先完成，**反向对照**（高先提交）则 **8/8** ⇒ **提交顺序主导、优先级无可观测效果**（差值 ≈ 提交间隔 1.2 ms）。**配额上限**：4k→256k 七档全成功、耗时线性；**100 万流** 13.93 s 成功、抽样 500/500 可用、宿主 **+30.8 MB（0.03 B/流）**、设备 **+10.0 MiB（0.010 B/流）** ⇒ **未观测到上限，且创建近乎零成本**（「配额未知」的真实原因不是探测不够）。过程如实留档：首版取数因**单次 matmul <2 ms 未形成争用**而无效（靠反向对照抓出）· 探针初版把范围三元组误判为不可读而**整段 SKIP**（「看起来通过」又一例）。见 `prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md` |
| ⭐ **C2 收尾：910C 宿主副本已对齐（09-30）** | ✅ **原路径 = 远端 tip、dirty=0；旧副本整目录留档** | 旧副本为 `856b24e`（v0.1.0 时代）· 落后 **208 提交** · **109 脏文件**。先做三重核对（**属主 / 时间 / 子树**）确认脏文件**全部是本方向自己的**（含路线 B 退出时的 `flagos/` 删除），**不含他人改动** ⇒ 才有资格对齐。处置：脏清单落盘 · 已跟踪 WIP 进 `stash`（可 `pop`）· 逐文件搬移撞上 **root 属主文件**（容器内建的）⇒ 改用**整目录留档 + 原路径新克隆**：旧副本 `runtime-team.stale-856b24e-20260930-hostcopy`（48 GB，含 stash 与备份目录，**一个文件没删**）。**照该副本操作不会再跑到旧代码** |
| ⭐ **B3 收尾：修订建议 10 条逐条状态复核（09-29 第八轮）** | ✅ **零僵尸 + 顺带修 2 处缺口 + 第 8 轮回归 9 项全绿** | 10 条三态：**已修 8 · 部分已修 1 · 未修 1**（第 6 条仅 smoke 打印未做；第 1 条 `vendor` 字段未加，但「拼不出有效设备串」的原始问题已随 `flagos` 退出而消解）⇒ 汇总表已回填为**现行三态**，不再有「已落地却仍标未改」的僵尸条目。过程中**另抓到并修 2 处真实缺口**：① `ascend.info()` **缺 `device_type`** ⇒ `info()["device_type"]` **KeyError**（另两家返回 `cuda`/`mlu`，且契约与 smoke 都要求该键）；② 三家 `info()` **各手写一份**公共字段、基类 `_base_info_fields()` **无调用方** ⇒ 同一份清单**三个来源**。修法：ascend 补键 + 三家并入基类唯一来源（**只增 / 等价迁移**）。新增 **2 条判据**（⭐ **对真实后端也守** —— 原能力矩阵取的是**类属性**、smoke 该判据只对 **stub** 跑 ⇒ 真实后端长期无人守）+ **非空转验证**。**第 8 轮回归 9 项全绿**：离线 **83/0/1** · 对称性 **7/0** · 冒烟 **52/0** · conformance 13+6 · 契约不变式 4/4（I1④ **14/14**）· 职责审计 **39/0/0** · B/C 探针 PASS · 推理腿 **`INFER_LEG_PASS 14/14`**（73.49 句/s）。见 `prototype/docs/INTERFACE_CONTRACT_REVISION_STATUS_REVIEW_20260929.md` |
| ⭐ **补公开入口：设备状态驱动 + 错误编排（09-29 第七轮续）** | ✅ **`ENTRY_VERIFY_PASS`** | 补 **2 个只增入口**（`runtime.set_device_state` 驱动四态 / `runtime.handle_error` R1–R5 编排）+ **1 个只增能力键** `device_state_control`（与 `device_state` 分开：**能查 ≠ 能改**）—— 原「某卡 L4 故障 → 设备级恢复」公开面上**不可触发**（无置隔离入口）已解决：真机 `handle_error(<L4 消息>, mode="real")` 端到端 `steps=['captured','evaluated: isolated','recovered: True','replay_ready']`；4 条新离线判据**全部非空转**；I1④「声明 ⇒ 有公共入口」覆盖 **13/13 → 14/14**；第 7 轮回归 **10 项全绿** ⇒ 见 `../prototype/docs/PUBLIC_ENTRYPOINTS_RECOVERY_20260929.md` |
| ⭐ **A2 收尾：多卡多进程 `real` 恢复压测（09-29 第七轮）** | ✅ **`MULTIPROC_REAL_RECOVER_PASS`** | 3 rank（一卡一进程）× **30 轮**、每 rank 各当 10 次恢复者 ⇒ **共 30 次真重建，零失败零异常**（11.0 s）· **S1** 五键 + context 三键取值正确（`context_recreated=True`）· **S2** 非恢复者 60/60 轮 digest **逐位不变**（89440）+ `device_state` 仍 `available` + `context_query` 快照不变 · **S3** 重建后再建流 + 状态回 `available`（R4）· **顺带挖出并修 3 处缺陷**（台账第 **21/22/23** 条，含 1 处「判据静默空转」）· 破坏面回归 **9 项全绿** ⇒ 报告 `docs/ASCEND_910C_A2_RECOVER_STRESS_20260929.md` |
| ⭐ **A1+A4 收尾：两条腿 + 服务化四形态（09-29 第六轮）** | ✅ **全绿 0 失败** | 训练腿 **`TRAIN_LEG_PASS 6/6`**（loss 15.4498→11.1479、**4513.2 tok/s**）· 推理腿前向 **`INFER_LEG_PASS 14/14`**（dim 1024、78.84 句/s、p50 37.59 ms、**区分度 0.6391**）· 服务化 **4 形态全 `SERVE_STANDARD_PASS`**（TP=1/2 × EAGER=1/0：就绪 30/45/45/80 s、冒烟 1/1/0/0 s、dim 1024、范数 1.000000；**TP=2 服务端实测 `world_size=2` + `Worker_TP0/TP1` + `backend=hccl`**；**EAGER=0 实测 `enforce_eager=False` + ACL Graph（PIECEWISE）**）· 顺手修掉 1 处**工具假信号**（停机复查把"正在回落"读成"未释放"，带非空转验证）⇒ 报告 `docs/ASCEND_910C_LEGS_SERVE_RERUN_20260929.md` |
| ⭐ **多卡上下文与语义细节（09-29 第五轮续）** | ✅ **三设备隔离成立** | 三设备各持独立上下文（互不相同），**销毁 dev0 后 dev1/dev2 仍非零** ⇒ 无交叉影响；⚠️ **`acl.rt.get_context(dev)` 的 `dev` 参数被忽略**（恒返回「当前设备的上下文」）⇒ 必须**先 `set_device` 再查**（本层实现已如此）；`managed_by` **能判出 `unified`**（建后 `unified`、销毁后 `present=False`）；IPC 入口 11 个存在（`ipc_mem_get_export_key` 等）⇒ 证据 `probes/PROBE_CONTEXT_probe_910c_multicard2.log` · `PROBE_CONTEXT_probe_910c_managed_by.log` |
| ⭐ **上下文只读观测 `context_query`（09-29 第五轮·对齐 P800）** | ✅ **真机 C4 PASS** | pyACL 只读入口实测：`acl.rt.get_context(dev)` 返回 **`(ctx, rc)` 元组**（3 次调用稳定一致）、`get_primary_ctx_state(dev)` 返回**三元组 `(1, 0, 0)`**（primary 已存在；设备 1 → `(0,0,0)`、越界 → `(0,0,107001)`）、**销毁后查询 `(0, 107002)` 如实报错**；`flags` **无此概念 ⇒ 如实 `None`**。真机 C4：`present=True` · `ordinal=0` · `managed_by="external"` · **`compute_before = compute_after = 512.0`（只读无副作用）** ⇒ 证据 `probes/probe_bc_contract_ascend_20260929_r3.json`（verdict 7 项全 true） | `allocate/free` 句柄（申请 8 MiB → 设备空闲 **−10.0 MB**）、二次释放如实 `ValueError`、`record_stream` 走**原生路径**、`.native` 审计隔离（公开 1 / 内部 0）、**上下文 create/set/destroy/count + 绑定语义在使用点拦截（`RuntimeError`）+ 多上下文隔离** ⇒ 报告 `../prototype/docs/WORKPACKAGE_BC_INTERFACE_20260929.md`；证据 `probes/probe_bc_contract_ascend_20260929.json` |
| ⭐ **共享层改动后的第 4 轮全套回归（09-29 r4）** | ✅ **10 项全绿 · 无回归** | 离线 **68/0/1** · 对称性 5/0 · 冒烟 **52/0** · conformance 13+6 · 职责审计 **39/0/0** · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3（本轮 `context_query` 在 ascend **尚未声明**；**已在「第五轮续」对齐** —— 现 ascend 已声明、真机 C4 PASS、离线判据数 **68 → 70**，见本表更上方一行）⇒ 证据 `probes/*_ascend_20260929_r4.*` |
| ⭐ **第 5 轮全套回归（09-29 r5）** | ✅ **11 项全绿 · 无回归** | 离线 **75/0/1**（判据数 68→75）· 对称性 5/0 · 冒烟 **52/0** · conformance 13+6 · **契约不变式 4/4（新增，`CONTRACT_INVARIANTS_PASS`）** · 职责审计 **39/0/0** · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 ⇒ 证据 `probes/*_ascend_20260929_r5.*`（22 份） |
| ⭐ **共享层改动后的第 3 轮全套回归（09-29 r3）** | ✅ **10 项全绿 · 无回归** | 离线 **64/0/1** · 对称性 5/0 · 冒烟 **52/0** · conformance 13+6 · 职责审计 **39/0/0** · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3（本轮唯一回归=冒烟 `info` 精确键集误报，已改判据）⇒ 证据 `probes/*_ascend_20260929_r3.*` |
| ⭐ **三处修复后全套回归（09-29 · 第 2 轮 r2）** | ✅ **10 项全绿、0 回归**：离线自检 **40/0/1 跳过** · 对称性 **5/0** · 冒烟 **52/0** · conformance **13/13 + 6/6** · 职责审计 **39/0/0** · 错误闭环 **5/0/0** · 等价性 **6/6** · 多流语义 **8/8** + 配额 **3/3**（第 1 轮 r1 为离线 **38/0/1**；本轮 +2 条「文案等价类」判据。**两轮证据都保留**）⇒ 见 `docs/ASCEND_910C_REGRESS_AFTER_FIX_20260929.md` |
| ⭐ **三芯片职责验收（09-22 傍晚 · 历史批次）** | ✅ **10 项全绿**：离线自检 **35/0** · 对称性 **5/0** · 冒烟 **52/0** · conformance **13/13 + 6/6** · 多流 语义 **8/8** + 图捕获 **4/4** + 配额 **3/3** · 训练腿 **`TRAIN_LEG_PASS 6/6`**（**4075.4 tok/s**）· 推理腿前向 **`INFER_LEG_PASS 14/14`** · 服务化 **`SERVE_STANDARD_PASS`** · 错误闭环 **`ERROR_RECOVERY_LOOP_PASS` 5/0/0** ⇒ 见 `../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` |
| 统一运行时 API + Backend 注册表 | ✅ 真机 **37/37** |
| 昇腾后端（torch_npu，**两条腿统一**） | ✅ conformance **13/13 + 6/6**、推理腿自验证 **10/10**、smoke **52/0** |
| 训练腿 2 卡分布式微调（Qwen3-Embedding-0.6B） | ✅ **现口径 = torch_npu + HCCL**：loss **15.4498 → 11.1479**（50 步）、**3954–4402 tok/s**、三类通信对照全对<br>⏹ torch_fl 线（历史）同模型同步数：**2212.9 tok/s**、loss 15.4497→11.1515 ⇒ 同口径下 torch_npu **+79~99%** |
| 推理腿单卡 · 前向形态 | ✅ 向量区分度 **0.638**、66–79 句/s、无 NaN |
| 推理腿单卡 · **服务化形态** | ✅ vLLM OpenAI 兼容服务 **10/10 SERVE_LEG_PASS**：维度 1024、区分度 0.4123、108 句/s（p50 27.4 ms）<br>✅ **统一脚本同形态验收（09-22）**：`SERVE_FORM=embed` + `Qwen3-Embedding-0.6B` ⇒ **`SERVE_STANDARD_PASS`**（35 s 就绪、维度 1024、范数 1.000000），与 P800 **同形态可比** |
| **错误注入 → 恢复闭环** | ✅ 推理腿 **5 闭环 / 0 失败**（含真实流同步超时 → L3_EXECUTION → 重放）；训练腿（torch_npu）**5 闭环 / 0 跳过 / 0 失败**；⏹ torch_fl 线（历史）4 闭环 / 1 跳过（无有界同步，如实跳过） |
| **统一启动脚本**（组内服务启动标准 v1.1，09-20） | ✅ **`SERVE_STANDARD_PASS (ready=1 smoke=1)`**：服务就绪 **30 s**；生成冒烟 **8 tokens**（`1+1=` → `'2 is a basic arithmetic fact, but'`）；用卡快照 `free=60.91GiB / total=61.27GiB`（停机前后一致）；宿主侧 8100 端口已释放。证据：`probes/L_serve_standard_910c_20260920.log` |
| 组件打包 | ✅ Git tag `runtime-v0.1.0` + Release note（`../prototype/RELEASE_NOTES_v0.1.0.md`） |

> **两条腿都是基于统一原型跑通的**（经 `runtime.use(...)` 接入设备），
> 脚本在 `../prototype/runtime/proto/`；**不是**把历史训推用原型重跑了一遍 ——
> 本目录 `distributed_training/` 的双卡 DDP（Qwen2.5-1.5B）与 `distributed_inference/` 的 vLLM+TP（Qwen3-4B）
> 属**旧代码路径**（直接 `import torch_npu` + `torch.distributed`，`runtime.use` 出现 0 次）。

---

## 4. 文档索引（本目录）

> **效力分层与全量文档说明见主看板 §6.4.1**；本节列出本实例的文档与证据，均为**实测记录**性质，
> 其中标注 ⚠️ 的条目**属于本实例专属结论，不迁移**给新芯片。

**核心：设备上下文与多流**

| 文档 | 一句话说明 |
|---|---|
| `ACL_ERROR_MAP_20260901.md` | ACL 错误码映射表建设记录（D10）：108 条码 → L1–L4 分级依据。⚠️ **不迁移** |
| `ASCEND_910C_DC_STREAM_MAPPING_20260902.md` | 设备上下文 × Stream 双侧全景（职责映射与实测）；**含 S-1～S-16 编号基线**（多流验收基线口径来源） |
| `DC_STAGE_SUMMARY_20260909.md` | 阶段性总结：两条腿证据并入 |
| `ERROR_RECOVERY_LOOP_20260909.md` | 错误注入 → 恢复闭环验证记录（含一处归因核查被推翻的记录） |
| `DIAG_TRAIN_IMAGE_NPU_20260908.md` | 训练腿锁定镜像 NPU 初始化失败排查记录 |
| `OFFICIAL_RUNTIME_COUNTERPART_20260922.md` | **FlagOS 官方对应镜像对照**：`flagos-runtime-ascend-cann9.0.0-910c:2.2.0` 与我们锁定栈**逐项一致**（CANN 9.0 / pt3.11 / torch 2.10 / triton 3.5 / **flagtree 0.7.0rc2+ascend3.5**）；**设备后端为 `npu`（Route A）** —— 与**我们已经统一的口径一致**（2026-09-22 起训练腿已走 torch_npu，路线 B 的权宜例外已取消）⇒ 本候选与现网差异**只剩镜像血统**。**仅登记，未切换** |
| ⭐ `docs/DUTY_RESPONSE_AUDIT_910C_20260928.md` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **39 OK / 0 FAIL / 0 SKIP**（三实例中唯一无 SKIP 者）；含本实例运行条件、逐项实测依据、复跑命令 |
| `docs/ASCEND_910C_REGRESS_AFTER_FIX_20260929.md` | **两处层内修复后的全套回归**：10 项判定全绿 / 0 缺陷；含两处修复的定向验证、工作包 A 实验取数、未跑项如实登记、一键复跑命令 |
| ⭐ `../prototype/docs/PUBLIC_ENTRYPOINTS_RECOVERY_20260929.md` | **公开入口补充记录**：`set_device_state` / `handle_error` / 能力键 `device_state_control` 的语义、只增性、4 条判据与**非空转验证**、端到端真机证据、第 7 轮回归 10 项、以及**未补的边界**（R5 在途登记入口 / 事件订阅未公开） |
| ⭐ `docs/ASCEND_910C_A2_RECOVER_STRESS_20260929.md` | **A2 收尾：多卡多进程 `real` 恢复压测**（当前原型）：3 rank × 30 轮、30 次真重建全绿；含压测设计（确定性整数 digest）、**3 处新缺陷的发现/修法/非空转验证**、破坏面回归 9 项、**免跑理由的当轮可验证形式**（腿=0 命中 / 服务=仅注释与 `MONITOR=1` 分支）、以及一条**未擅自改的接口面事实**（公开面无 ISOLATED 入口）· 一键复跑 |
| ⭐ `docs/ASCEND_910C_LEGS_SERVE_RERUN_20260929.md` | **A1+A4 收尾：两条腿与服务化四形态复跑**（当前原型）：训练腿 `6/6` · 推理腿 `14/14` · 服务化 TP=1/2 × EAGER=1/0 全 `SERVE_STANDARD_PASS`；含「为什么上一轮的免跑理由失效」的破坏面依据、TP 与图捕获的**生效证据**、1 处工具假信号的修复与**非空转验证**、一键复跑 |
| ⚠️ `docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md` | **宿主带卡容器名额规则判别实验与口径更正**（5 数据点）：独占单位 = 已 init 容器的**挂载设备集**；「同一时刻只留 1 个」「与挑哪张卡无关」两条旧口径均**不成立**；⚠️ **不迁移**（宿主专属） |

**8 月早期工作（FlagCX 补丁与准备）**

| 文档 | 一句话说明 |
|---|---|
| `DEVICE_CONTEXT_PLAN_20260827.md` | 设备执行上下文方案定稿（8 月版） |
| `PROGRESS_20260822.md` | 8-22 阶段进度快照 |
| `910C-env-issue-report.md` | 容器内 `aclInit` 返 500000 记录 —— ⚠️ 该文所记「**根因 = DrvMng 容器上限 3**」**已作废**（现行口径见 §5 与 `docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`）⚠️ 环境专属 |
| `O3_getlasterror_fix.md` ｜ `O4_socket_seq_guard.md` | FlagCX O3/O4 缺陷修复设计与实现 |
| `PR_DEV_1_0_20260902.md` | PR #11 合入 dev-1.0 记录（157 文件） |

**训练 / 推理既有工作（旧代码路径，非统一原型）**

| 目录 | 一句话说明 |
|---|---|
| `distributed_training/docs/` | 5 份：训练 × 设备上下文映射与验收评估、FlagCX 核心缺陷修复（死锁 P2 / 数据错乱 P6 / OOM P7）、net.cc chunk 竞态调研、A 线验证、4090 报告 |
| `distributed_inference/docs/` | 5 份：推理 × 设备上下文映射与验收依据、实现方案、P0/P1 执行记录、P3 服务化 × 状态/错误恢复（A8–A10）、Qwen3-4B TP 数值等价性（⚠️ 须 greedy） |

**证据（`probes/`）**

| 证据 | 内容 |
|---|---|
| `recheck_conformance_13_ascend_20260922.json` | 一致性判据 13 例（含 `backend`） |
| `recheck_conformance_infer6_ascend_20260922.json` | 一致性判据 推理 6 例 |
| `recheck_stream_semantics_ascend_20260922.json` | 执行语义基线 8/8（后端无关 V2 探针**首跑 ascend**） |
| `recheck_error_loop_ascend_20260922.json` | 错误闭环 闭环 5 / 跳过 0 / 失败 0（含 `backend` + 时间戳） |
| `recheck_infer_leg_ascend_20260922.json` | 推理腿前向 **14/14**（含 `backend` + `env` + p50/p90） |
| `recheck_train_leg_ascend_20260922_rank{0,1}.json` | 训练腿 2 卡 **6/6** —— ⚠️ 该次 `backend=flagos`（torch_fl），**属切换前的旧口径证据**（loss 15.4497→11.1515、**2212.9 tok/s**） |
| `train_npu_20260922.log` | ⭐ **切换后**训练腿（**torch_npu + HCCL**）：`TRAIN_LEG_PASS 6/6`、loss **15.4498→11.1479**、**4402.3 tok/s** |
| `unified_verify_20260922.log` | ⭐ **统一口径完整复核**：离线自检 35/0 · smoke **52/0** · conformance **13/13 + 6/6** · 训练腿 **6/6**（3954.0 tok/s） · 错误闭环 **5/0/0** · `--all` 5/0 |
| `ev_matrix_20260922.log` / `ev_matrix2_20260922.log` | torch_fl `Event.query()` 语义缺口实测矩阵（审计台账第 13 条的证据） |
| `L_serve_standard_910c_20260920.log` | 《组内服务启动标准》脚本真机验证日志（`SERVE_STANDARD_PASS ready=1 smoke=1`） |
| ⭐ `accept_*_20260922.{log,json}`（15 份） | **三芯片职责验收全套证据**（09-22 傍晚）：离线自检 · 对称性 · 冒烟 · conformance 13/13 与推理 6/6 · 三个多流探针 · 训练腿（`accept_train_npu_20260922/`，含两 rank JSON）· 推理腿前向 · 服务化 · 错误闭环；另有 `accept_probe_results_910c_20260922/`（探针原始 JSON） |
| ⭐ `accept_serve_ascend_*_20260928.{log}`（3 份） | **服务化按新脚本（v1.2，含 `SMOKE_TIMEOUT`）复跑**：`SERVE_STANDARD_PASS`（就绪 **35 s**、维度 1024、范数 1.000000、**冒烟耗时 0 s**），与 09-22 逐项一致 |
| `accept_serve_ascend_*_20260928_NAMESLOT_BLOCKED.log`（3 份） | **同轮首跑失败证据（原样留档，未「改判据变绿」）**：宿主带卡容器名额被他人占满 ⇒ `acl.init`=500000、`get_device_count`=(0,0)、vLLM `Engine core initialization failed`（root cause 原文 `Failed to obtain the console log level … Different containers share the same device`） |
| ⭐ `duty_audit_ascend_20260928.json` | **职责响应审计（39 项 sub-part × 真机）**：`DUTY_RESPONSE_PASS` **39 OK / 0 FAIL / 0 SKIP**；补做后回归复跑仍 39/0/0（无退化） |
| ⭐ `regress_*_ascend_20260929.{log,json}`（11 份） | **缺陷修复后全套回归（真机）**：离线自检 38/0/1 · conformance 13/13 与 6/6 · 职责审计 39/0/0 · 错误闭环 5/0/0 · 冒烟 52/0 · 多流语义 8/8 · 流配额 3/3 |
| ⭐ `regress_*_ascend_20260929_r2.{log,json}`（15 份）+ `exp_divergence_cost_ascend_20260929_r2.{json,log}` | **第 2 轮（含第三处 L2 文案等价类修复）全套回归（真机）**：离线 **40/0/1** · 对称性 5/0 · 冒烟 52/0 · conformance 13+6 · 职责审计 39/0/0 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⭐ `probe_bc_contract_ascend_20260929.json` | **工作包 B/C 真机契约探针**（5 组逐组独立子进程）：B1/B3/B4/C1/C2/C3 **6/6 通过**；含 allocate→占用→free 的设备空闲变化、二次释放负向、**厂商原生流「销毁后使用 = 静默成功」的对照取证**（本层则如实拦截） |
| ⭐ `regress_*_ascend_20260929_r5.{log,json}`（22 份）+ `recheck_*_r5b.json` | **第 5 轮全套回归（当前原型 · 11 项全绿）**：离线 **75/0/1** · 对称性 5/0 · 冒烟 52/0 · conformance 13+6 · **契约不变式 4/4** · 职责审计 39/0/0 · 错误闭环 5/0/0 · 等价性 6/6 · 多流 8/8 + 配额 3/3 |
| ⭐ `legs_train_910c_npu_20260929.log` + `train_leg_910c_npu_20260929_rank{0,1}.json` + `legs_infer_910c_npu_20260929.log` + `infer_leg_910c_npu_20260929.json`（5 份） | **A1 两条腿复跑（当前原型）**：训练腿 `TRAIN_LEG_PASS 6/6`（loss 15.4498→11.1479、**4513.2 tok/s**）· 推理腿前向 `INFER_LEG_PASS 14/14`（dim 1024、78.84 句/s、p50 37.59 ms、区分度 0.6391） |
| ⭐ `serve_standard_910c_npu_20260929_r1_tp{1,2}_eager{0,1}.log`（4）+ `serve_vllm_910c_npu_20260929_r1_tp{1,2}_eager{0,1}.log`（4）+ `serve_pool_910c_npu_20260929_r1.log` + `serve_standard_910c_npu_20260929_r1_nonidle_wait5.log`（10 份） | **A1+A4 服务化四形态（当前原型）**：TP=1/2 × EAGER=1/0 **全 `SERVE_STANDARD_PASS`**；TP=2 服务端日志实测 `world_size=2` + `Worker_TP0/TP1` + `backend=hccl`；EAGER=0 实测 `enforce_eager=False` + ACL Graph（PIECEWISE）；`nonidle_wait5` 为**非空转验证**（证明 `⚠️` 释放复查分支能真的触发） |
| ⭐ `a2_recover_multiproc_910c_npu_20260929.{json,log}`（2）+ `a2p1_fix_verified_910c_npu_20260929.{json,log}`（2）+ `a2_p0_public_surface_910c_npu_20260929.{json,log}`（2）+ `r6_regress_910c_npu_20260929.log` + `r6_regress_910c_npu_20260929_out/`（16）+ `offline_3backends_910c_npu_20260929.log` | **A2 全套证据（24 份）**：压测逐轮原始结果（3 rank × 30 轮 digest/快照/返回 dict/状态机转换）· A2-P1 修复双向定向验证 · A2-P0 公开面事实探针 · 第 6 轮破坏面回归 9 项 + 免跑理由验证 · 离线自检三家同跑（78/80/68 全 0 失败） |
| ⭐ `r8_regress_910c_npu_20260929.log` + `r8_regress_910c_npu_20260929_out/`（15 份） | **第 8 轮全套回归（当前原型 · 9 项全绿）**：破坏面 = `info()` 返回值（三家 `backend.py`）⇒ **元信息面**改动。离线 **83/0/1** · 对称性 **7/0**（含新增 2 条 `info()` 判据）· 冒烟 **52/0** · conformance **13/13 + 6/6** · 契约不变式 **4/4**（I1④ 入口存在性覆盖 **14/14**）· 职责审计 **39/0/0** · B/C 探针 **PASS** · 推理腿 **`INFER_LEG_PASS 14/14`**。**免跑项（实测理由）**：服务化（`serve_standard.sh` 仅 1 处注释命中 `device_type`，非真实调用）· 训练腿（`grep -c "\.info()" runtime/proto/proto_train_leg.py` = 0） |
| ⭐ `entry_verify_910c_npu_20260929.{json,log}`（2）+ `a2_recover_multiproc_910c_npu_20260929_pubentry.{json,log}`（2）+ `r7_regress_910c_npu_20260929.log` + `r7_regress_910c_npu_20260929_out/`（17）+ `offline_3backends_910c_npu_20260929_r7.log` | **补公开入口轮证据（23 份）**：入口定向验证（D1–D4 含 `handle_error` 端到端）· 压测改用公开入口后复跑 · 第 7 轮破坏面回归 10 项原始输出 · 离线三家同跑（83/85/73 全 0 失败） |
| ⚠️ `serve_standard_910c_npu_20260929_PRE_FIX_tp{1,2}_eager{0,1}.log`（4）+ `serve_vllm_910c_npu_20260929_PRE_FIX_tp{1,2}_eager{0,1}.log`（4） | **改动前原样留档**（未「改判据变绿」）：TP=2 停机后即时复查读 dev0 **55.06 / 55.07 GiB**（用卡前 61.12）⇒ 会被误读为「未释放」，实为**释放延迟**（约 1 min 后回基线，宿主 `npu-smi info -t proc-mem` 全程 `No process`） |
| ⭐ `exp_divergence_cost_ascend_20260929.{json,log}` | **工作包 A 实验的 910C 取数**：M1–M4（0/0/77/0 vs 11/15/163/17）· 功能等价性 **6/6 一致**（`S6` = `L4_FATAL`，ascend 声明了 `error_map` ⇒ 正确） |
| ⚠️ `nameslot_rule_matrix_ascend_20260929.log` | **宿主名额规则判别实验原始留档**（2×2+1 五个数据点，原样输出）：`davinci7` 成功 / `davinci1`、`davinci2` 失败 ⇒ 旧口径「与挑哪张卡无关」被推翻 |

> 证据命名规范（批次 / 条件 / 日期）与「当前结论 = 哪一份」见 `../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5。

**分支看板**

- `distributed_training/README.md` ｜ `distributed_inference/README.md`

---

## 5. 环境要点（910C 专属，不迁移）

- ⚠️ **带卡容器名额（2026-09-29 口径更正）**：宿主的独占单位是「**已初始化 ACL 的容器的挂载设备集**」——
  某容器一旦真正 `acl.init` 成功，就**把它挂载的全部设备整组独占**；其他容器只要与之**挂载集有交集**，
  `acl.init()` 即失败。**空闲（`Up` 但未 init）的带卡容器不占名额**（实测 5 个 Up 带卡容器并存无碍）。
  ⇒ **可操作做法：起容器时不要挂全 16 张**（全量挂载几乎必然与任何活跃者相交 ⇒ 必失败），
  **只挂自己真正要用的空闲卡**，即可与既有活跃容器并存，**不必再停用他人容器**。
  真机症状：`acl.init()` 返 **500000**、`get_device_count()` 返 **(0, 507899)**、stderr 首条
  `Failed to obtain the console log level` + `Different containers share the same device`（vLLM 侧表现为
  `Engine core initialization failed`）；**而 `npu-smi info` 仍报 `Health: OK`** ⇒ **芯片健康 ≠ 名额有空**。
  处置：**先查自己的挂载集与活跃容器挂载集是否相交**，不要先怀疑镜像/驱动/代码，
  **也不要在代码里加重试**（宿主占用约束，重试无用）。
  ⚠️ **历史口径两次更正**：① `dev/stack.lock.910c.v2.yaml` 与早期文档记的「DrvMng 名额 **≈3**」**到不了 3**；
  ② 09-28 由「起临时容器只挂 `davinci0` 仍失败」推出的「**与挑哪张卡 / 卡是否重叠无关**」**不成立** ——
  `davinci0` 正是 `evalx-910c` 的挂载设备，那次失败可由"与某活跃容器挂载集相交"解释；
  09-29 同宿主同刻实测：挂 `davinci7` **成功**（`acl.init rc=0`）、挂 `davinci1` / `davinci2` **均失败**
  ⇒ **选卡（挂载集是否相交）就是决定因素**。完整实验（5 个数据点）见 `docs/ASCEND_HOST_NAMESLOT_RULE_20260929.md`。
  两条腿**仍须串行**（训练容器与推理容器都挂全部 16 个 `davinci`，互斥必然）。
  另：**训练容器自带 `vllm` 入口但缺包**（`command -v vllm` 有、`import vllm` 报 `ModuleNotFoundError`）
  ⇒ 服务化**必须用推理容器**（`flagos-infer-910c`，或**镜像相同、只挂要用的卡**的等价精简容器
  `dc-lean-infer-910c-20260929` —— 本轮即用它只挂 `davinci1,2`，避免与任何活跃容器挂载集相交）；`serve_standard.sh` 的"找不到 vllm 就激活 conda"兜底对 910C 不适用。
- **训练镜像** `flagrt/ascend-operator-runtime-comm:0.1.3`（**镜像未变**）
  ⇒ **2026-09-22 起训练腿改走 `npu`（torch_npu）**：用容器内**带 torch_npu 的解释器**
  `/mnt/raid/hliu553/venvs/venv-infer-a/bin/python`（该解释器**无 torch_fl** ⇒ **物理隔离**，
  不触发镜像那条"禁止共存"校验）；`DC_BACKEND=ascend DC_DIST_BT=hccl`。
  ⚠️ **必须在 `init_process_group` 之前先经后端触碰一次设备**，否则 `hccl` 未注册，
  报 `AssertionError: Unknown backend type hccl`（审计台账第 15 条）。
  ⏹ 原路线 B 走法（`AUTOLOAD=0` + 先导入该插件）**已随路线 B 归档**：原型里的该后端已删除，
  当前口径不需要它，也不再提供该路径。
- **推理镜像** `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` → 后端 **npu（torch_npu）**。**两腿现已同后端**。
- 选卡变量：`ASCEND_RT_VISIBLE_DEVICES`。
- **容器内没有 `npu-smi`**（实测 `npu-smi: command not found`）——它是**宿主工具**。
  容器内查卡请退回 torch 侧（`torch.npu.mem_get_info`）；要看整机 16 卡全貌在宿主执行 `npu-smi info`。
  统一启动脚本已按此降级（`[torch.npu:0] free=… / total=…`）。
- **⭐ FlagOS 官方已有 910C 专用镜像（2026-09-22 登记，未切换）**：
  `harbor.baai.ac.cn/flagos-runtime/flagos-runtime-ascend-cann9.0.0-910c:2.2.0`
  （digest `sha256:1048d622…`，5.4 GiB，官方标**宿主驱动前置 26.0.rc1**，实测可匿名拉取）。
  与我们锁定栈**逐项一致**，但**不含 FlagCX** ⇒ 训练腿不能直接替代；
  见 `docs/OFFICIAL_RUNTIME_COUNTERPART_20260922.md`（含逐项对照与切换前置条件）。
  另：其 `base|runtime/<backend>.md` **明示宿主驱动前置**，建议我们 `dev/images/<name>/v<N>/lock.yaml` 吸收该字段。
- **起服务统一走《组内服务启动标准》**：`../prototype/scripts/serve_standard.sh`（唯一入口，
  `DC_BACKEND=ascend`）。本目录的 `distributed_inference/inference/start_vllm_serve_910c.sh`
  含 D10/D11 集成（错误翻译包装器 + 设备状态监控），**保留但仅供该集成场景**，下游新需求请走统一脚本。
