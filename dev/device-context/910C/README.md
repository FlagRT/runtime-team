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
| ⭐ **D2 调度效果对照实验（10-08 · 第十三轮续补跑）· 最新** | `NOT_APPLICABLE`（**如预期**）· `instrument_valid=True`（工作量级 **13.651 ms** · 分辨力 **0.021%** · 正对照 **8/8 / +12.75 ms**）。本实例**区间可读** `(7,0)` 但**未声明 `control`** ⇒ 与 P800 的「单档」是**两种不同的不适用** | `../prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L123–L150**（§4.5） |
| ⭐ **台账 E1：职责审计扩到 78 项（10-08）** | `DUTY_RESPONSE_PASS` **73 / 0 / 5**（SKIP 均如实不具备）· 非空转 **34 抓到 / 5 不适用 / 0 未抓到** · 顺带修 2 处真缺陷（统一面缺 `context_set` / `stream_priority_range` 出口） | `../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92**（§二 · §三·补） |
| ⭐ **r10 第 10 轮复跑（10-08 · 台账 D4 收尾）** | **20 项全绿** · 离线 **90/0/1** · 冒烟 52/0 · conformance 13+6 · **两条腿 6/6 + 14/14** · 服务化 **4 形态全 PASS** · `MULTIPROC_REAL_RECOVER_PASS` | `docs/ASCEND_910C_R10_RERUN_20261008.md` **L10–L49**（§0）· 两个偏离项 **L89–L222** |
| ⭐ **「不适用」自证审计（10-08）** | 结论**成立**，且根因**升级**：**不是设备限制** —— ACL 侧 `aclrtCreateStreamWithConfig(0/3/7)` 全部保留；是**插件 Python 绑定缺失**（C++ 已有 `getStreamFromExternal`） | `../prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L46–L62**（§2.1 逐条读数） |
| 流优先级统一 API 落地（09-30 第九轮） | `STREAM_PRIORITY_API_PASS` **8/8**；本家**如实结论 = 只读**（范围 `(7,0)` 可读、**不声明设置**） | `../prototype/docs/STREAM_PRIORITY_API_20260930.md` **L152–L169**（§4.3 六条证据） |
| B2 收尾：流配额 + 优先级（09-30） | **真实配额上限 1979**（pyACL 建满 1979、第 1980 条 `rc=207005`）；优先级**在插件层丢参数**（C API 回读恒 0） | `../prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md` |
| A1+A4：两条腿 + 服务化四形态（09-29 r6） | 训练腿 **6/6**（4513.2 tok/s）· 推理腿前向 **14/14** · 服务化 TP=1/2 × EAGER=1/0 **全 PASS**（TP=2 实测 `world_size=2`+`backend=hccl`） | `docs/ASCEND_910C_LEGS_SERVE_RERUN_20260929.md` **L45–L142** |
| A2：多卡多进程 `real` 恢复压测（09-29 第七轮） | `MULTIPROC_REAL_RECOVER_PASS`（3 rank × 30 轮 · **30 次真重建零失败**） | `docs/ASCEND_910C_A2_RECOVER_STRESS_20260929.md` |
| 公开入口：`set_device_state` / `handle_error`（09-29 第七轮续） | `ENTRY_VERIFY_PASS`（端到端 R1–R5）+ 能力键 `device_state_control` | `../prototype/docs/PUBLIC_ENTRYPOINTS_RECOVERY_20260929.md` |
| 工作包 B/C：内存句柄 + 上下文生命周期（09-29） | 真机 **6/6**（`allocate/free` 句柄、`record_stream`、`.native` 审计隔离、上下文四件事） | `../prototype/docs/WORKPACKAGE_BC_INTERFACE_20260929.md` **L24–L40 / L127–L167** |
| 训练腿 / 推理腿 / 错误闭环（基本面） | 训练腿 **6/6**（loss 15.4498→11.1479）· 推理腿 **14/14**（区分度 0.6391）· 错误闭环 **5/0/0** | `docs/DC_STAGE_SUMMARY_20260909.md` · `docs/ERROR_RECOVERY_LOOP_20260909.md` |
| 三芯片职责验收（09-22 傍晚 · 历史批次） | 10 项全绿（离线 35/0 · 冒烟 52/0 · conformance 13+6 · 两条腿 · 服务化 · 错误闭环） | `../prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` |
| 统一运行时 API + Backend 注册表 | ✅ 真机 **37/37** | `../prototype/docs/RUNTIME_PROTOTYPE_DESIGN_20260904.md` |
| 组件打包 | Release **`runtime-v0.2.0`**（三芯片统一原型版）；本实例首版 `runtime-v0.1.0` | `prototype/RELEASE_NOTES_v0.1.0.md` · `prototype/RELEASE_NOTES_v0.2.0.md` |

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

**证据（`probes/`）** —— 全量清单（数十组、上百份原始日志与 JSON）已移入
[`docs/EVIDENCE_INDEX_910C.md`](docs/EVIDENCE_INDEX_910C.md)；本看板只留入口：

| 用途 | 证据入口 |
|---|---|
| 当前结论 = 哪一份 | `probes/r10_*_20261008*` · `probes/e1_regress_910c_npu_20261008_out/` · `probes/d2_20261008_out/` · `probes/audit_20261008_out/` |
| 两条腿 / 服务化 / 压测 | `probes/legs_*_20260929*` · `probes/serve_standard_*_r1_*` · `probes/a2_*_20260929*` |
| 回归批次 | `probes/r8_/r9_/r10_*` · `probes/regress_*_r{2,3,4,5}.*` · `probes/recheck_*_20260922.json` |
| 口径与命名规范 | `../prototype/docs/VERIFICATION_MANIFEST_20260920.md` §2、§5 |

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
