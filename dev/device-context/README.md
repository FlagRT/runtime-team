# 设备上下文（device-context）· 主看板

> 分支：`kistich/device-context` ｜ PR 目标：`dev-1.0` ｜ 更新：**2026-10-10**
> 职责：**运行时层的设备抽象与执行上下文**（上承算子层/编译层，下接多机多卡分布式训练推理）
> 统一基座配置：**`dev/stack.lock.910c.v2.yaml`**（总组定稿；锁定镜像、使用规则、两条腿约束）
> 本方向状态文件：**`dev/device-context/STATUS.md`**（按全组约定，总组据此收拢诉求与裁定基座）
> 基座工作草稿：**`dev/stack.lock.910c.yaml`**（本方向工作副本，留在特性分支不直接 PR；
> **每周三**按设备上下文与多流 Stream 进展评估：无更新以总组 v1 为准，有更新写入草稿并同步 STATUS.md）
>
> 📖 **下游子方向请先读 §6「文档总目录」**——按效力分层（规范/效力 · 操作手册 · 参考记录）给出每份文档的一句话说明
> 与四条阅读路径；**要按统一接口对接读 §6.1，要起服务 / 接新芯片 / 做复核读 §6.2**。
>
> 当前状态：**三实例（910C / P800 / MLU590）职责验收 12 项全部通过，原型可发布** ——
> 厂商 torch 插件统一原型下，同一套判据 / 同一份脚本逐芯片复跑：离线自检 · 跨后端对称性 · 冒烟 ·
> conformance 13/13 + 6/6 · 多流 16 项 · 训练腿 6/6 · 推理腿前向 · 推理腿服务化（`SERVE_STANDARD_PASS`）·
> 错误注入→恢复闭环，**三实例均无失败项**（矩阵见 `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md`）。
>
> ⭐ **2026-10-08 更新**：三实例已在**同一代码版本**上跑完**扩口径后的 78 项职责审计** ——
> **910C `73/0/5` · P800 `67/0/11` · MLU590 `61/0/17`**（SKIP 均为**如实不具备**），
> 并各配**逐条注入的非空转验证**（新增「无需设备」的 §0 委派翻译层自检）。
> **台账 D1 已收尾**（MLU590 是三家唯一「能设置流优先级」的实例，在新接线下真机通过 7/7）；
> **D2** 为可选的效果对照，**E2** 为新增的覆盖面登记项（三家现役实例都行使不了契约 §1.10 的 L5）。
>
> ⭐ **2026-10-10 更新**：**第 4 家实例（平头哥 PPU）已上机**并取得**同口径的两条腿判据** ——
> 接入（离线自检 **112/0/1** · 对称性 **7/0** · 冒烟 **46/0** · conformance **13/13 + 6/6**）+
> **步 0 模型完整性 PASS** + **训练腿 50 步 6/6**（5167 tok/s）+ **推理腿 13/13**（135.77 句/s）；
> **同日同口径自审**补齐 6 个标准步骤（多流 8/8 · 图捕获 4/4 · 配额 3/3 · B/C 7/7 · 环境普查 · 职责审计），
> 当场抓出 **职责审计 1 条 FAIL（`L1` `release_stream` 越权销毁）** —— ✅ **根因已定位并修复**（跨类型地址复用 ⇒ 类型复核，机制无关；弱引用下亦 3/3 OK），
> 并更正首轮训练腿 20 步的口径问题 ⇒ `PPU/docs/PPU_ONBOARDING_PARITY_AUDIT_20261010.md`。
> ⚠️ 两条腿均在**判据通过后于解释器退出阶段段错误**（与 `runtime.create_stream()` 强相关，**归属未定**）；
> ⬜ 第 4 家**尚未**跑：多流 16 项 · 服务化 · 错误闭环 · 职责审计 78 项。
> ⇒ 逐项读数见 §2.6；报告 `PPU/docs/PPU_MODEL_AND_TWO_LEGS_20261010.md`。

---

## 1. 目录结构：**一份原型 + 四个芯片实例**

| 目录 | 定位 | 内容 |
|---|---|---|
| **`prototype/`** | **芯片无关的统一标准**（我们制定的规范、实现与判据） | 统一运行时 API、Backend 注册表与后端实现（`ascend` / **`kunlun`** / `cambricon` / **`ppu`**）、conformance 用例与 runner、两条腿自验证脚本、**接口约定与事件语义契约**、原型设计、月度计划、**职责验收报告**（`prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` 与 **09-28 三芯片矩阵**） |
| **`910C/`** | **第一个落地实例（昇腾）** —— ✅ 已完成 | 分布式训练与推理既有工作、910C 专属文档（ACL 错误码表、双侧映射、阶段总结、错误闭环、镜像诊断） |
| **`P800/`** | **第二个接入实例（昆仑芯）** —— ✅ 阶段 0–5 完成 | 环境汇总、五域基线、接入方案、根因核对、阶段 3/4 与镜像等价性验证、全量进度报告、**探针脚本与原始证据** |
| **`MLU590/`** | **第三个接入实例（寒武纪）** —— ✅ **接入完成 · 12 项判定全部通过（2026-09-28）**（环境打通 → 镜像定档 → 后端落地 → conformance 13+6 → 16 项基线 → 训练腿 → **推理腿前向 + 服务化** → 错误闭环） | 环境报告、镜像渠道调研与更正、**接入方案 + 真机执行手册**、**16 项基线比对报告**、探针脚本与原始证据（`docs/`、`probes/`） |
| **`PPU/`** | **第四个实例（平头哥 PPU / 真武 ZW810E）** —— 🟢 **接入自检 + 两条腿已达三家同口径（2026-10-10）**（离线 **112/0/1** · 对称 **7/0** · smoke **46/0** · conformance **13/13 + 6/6** · 步 0 模型完整性 PASS · **训练腿 50 步 6/6** · **推理腿 13/13** · 多流 **8/8** · 图捕获 **4/4** · 配额 **3/3** · B/C **7/7**）；⚠️ **未做：服务化 / 错误闭环 / 非空转**；✅ **职责审计 72/0/6（`L1` 根因已定位并修复）**；⚠️ 两条腿判据通过后**退出期段错误**（归属未定） | **同口径自审报告** · 接入报告（能力逐项实测表 + 3 处既有工具缺陷）· **模型与两条腿报告**（含 16 组段错误对照）· 接入前调研（含复核）· **4 轮能力探测探针** · 两轮验证日志 |

顶层保留：`README.md`（本看板）、`STATUS.md`（方向状态）、`.env.example`
各有分支看板：`prototype/README.md`、`910C/README.md`、`P800/README.md`、`MLU590/README.md`、`PPU/README.md`

> **划分原则：原型与规范是芯片无关的，放在外面；芯片专属的落地实例资产按芯片分目录。**
> 新芯片接入 = 在 `prototype/` 下**新建一个 backend** + 跑通 conformance，
> **不复制任何既有芯片的实现**（详见 `910C/README.md` §1 铁律）。

---

## 1.1 分支与 dev-1.0 的关系

| 项 | 状态 |
|---|---|
| PR #11 | 已于 **2026-09-02 合入 dev-1.0**（157 文件），当时为旧扁平结构（`benchmarks/`） |
| 本分支在此后 | ① 仓库重组 ② 新增统一原型 `prototype/` ③ 基于原型的训推复跑 ④ 错误注入→恢复闭环 ⑤ **P800 第二实例接入** ⑥ **按芯片重组目录（原型 + `910C/` + `P800/`）** —— ⑤⑥ 已随 **PR #19** 合入 |
| **PR #19** | 已于 **2026-09-17 合入 dev-1.0**：**P800 第二实例接入 + 按芯片重组目录**（一份原型 + 两个芯片实例），241 文件（+6591/−179，GitHub 重命名识别后口径） |
| 当前相对 dev-1.0 | **领先 41 个提交**（09-22 复核实测；含 PR #19 后的今日批次）：P800 阶段 3/4（推理腿 13/13 + 服务化 10/10 + 错误闭环两设置对照）、**官方 `-base` 镜像等价性验证**、吞吐差异拆解、两实例配置手册、镜像需求说明书与选择指南 |
| 下一轮合入 | 按 `dev/stack.lock.910c.v2.yaml` 的**合入把关五条**走流程（P800 阶段 5 收敛完成后） |

---

## 2. 当前状态总览

### 2.1 原型（芯片无关）

| 项 | 状态 | 证据 |
|---|---|---|
| 统一运行时 API + Backend 注册表 | ✅ | `prototype/runtime/`，真机 37/37（910C）/ 42/0（P800） |
| Backend 抽象与接入规范 | ✅ | 13 个 `@abstractmethod` 对应五域；`registry._KNOWN_BACKENDS` 已含 `ascend` / `kunlun` / `cambricon` |
| **《新芯片接入手册》** | ✅ | `prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`：判别路径 · 13 抽象清单 · 8 步流程 · **验收清单** · 跨芯片坑 · 上报模板 |
| **《组内服务启动标准》** | ✅ | `prototype/scripts/serve_standard.sh`（**下游起服务唯一入口**，跨芯片只改 `DC_BACKEND`）+ `prototype/docs/SERVICE_STARTUP_STANDARD_20260920.md`；**910C / P800 真机均 `SERVE_STANDARD_PASS`**（910C 就绪 30 s + 生成冒烟 8 tokens；P800 就绪 25 s + 冒烟维度 1024）；**09-22 三实例同形态验收**（新增 `SERVE_FORM=embed`）—— 910C 35 s、P800 25 s 就绪，均 PASS；**09-29 新增 `RELEASE_WAIT`**（停机后释放回落等待上限，默认 30 s，轮询到稳定再打印「用卡后复查」，**不参与 verdict**） |
| **环境普查脚本** | ✅ | `prototype/scripts/preflight_env.sh`（接入手册 §1 的 7 项可执行化）；第 3 家接入的第 0 步，产出即环境报告 |
| **《验证复核清单》** | ✅ | `prototype/docs/VERIFICATION_MANIFEST_20260920.md`：**10 条**"声明 → 命令 → 判据"最小复现表 · 证据索引（含"当前结论 = 哪一份"）· **缺口 G1–G8** · 复跑阻塞项 · 证据命名规范 |
| conformance 判据集 | ✅ | 功能 13 例 + 推理 6 例，三个后端结果并列可比 |
| 组件打包 · **最新** | ✅ | **GitHub Release [`runtime-v0.3.0`](https://github.com/FlagRT/runtime-team/releases/tag/runtime-v0.3.0)**（**2026-10-08**：接口面补齐 §1.6–§1.10 + 8 处行为修正，含附件 `runtime-prototype-v0.3.0.tar.gz`）+ `prototype/RELEASE_NOTES_v0.3.0.md`；上一版 `runtime-v0.2.0`（三芯片统一原型版）、更早 `runtime-v0.1.0`。⚠️ **版本以 Release 为准**（git tag 仅作提交指针） |

### 2.2 910C 实例（第一个落地实例，✅ 已完成）

| 项 | 状态 | 证据 / 详见 |
|---|---|---|
| 后端（`ascend` = torch_npu，**两条腿统一**） | ✅ conformance **13/13 + 6/6** · 冒烟 **52/0** · 离线自检 **90/0/1** | `910C/README.md` §3 · `prototype/scripts/backend_offline_check.py` |
| 两条腿 + 服务化 | ✅ 训练腿 **6/6**（loss 15.4498→11.1479 · 4196.6 tok/s）· 推理腿 **14/14**（区分度 0.6391）· 服务化 **4 形态全 `SERVE_STANDARD_PASS`** | `910C/docs/ASCEND_910C_LEGS_SERVE_RERUN_20260929.md` **L45–L142** · `910C/docs/ASCEND_910C_R10_RERUN_20261008.md` **L10–L49** |
| 错误注入 → 恢复闭环 | ✅ 推理腿 **5 闭环 / 0 失败** · 训练腿 **5/0/0** | `910C/docs/ERROR_RECOVERY_LOOP_20260909.md` |
| ⭐ **职责审计（78 项口径 · 10-08）** | ✅ `DUTY_RESPONSE_PASS` **73 / 0 / 5**（SKIP 均如实不具备）· 非空转 **34 抓到 / 5 不适用 / 0 未抓到** | `prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92** |
| ⭐ **流优先级** | ⚪ **只读**：范围 `(7,0)` 可读、**不声明设置**（`create_stream(priority=…)` ⇒ `NotImplementedError`）；「调度效果」如实 `NOT_APPLICABLE` | `prototype/docs/STREAM_PRIORITY_API_20260930.md` **L152–L169** · `prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L123–L150** |
| ⭐ **「不适用」自证审计（10-08）** | ✅ 结论成立、**根因升级**：**不是设备限制**（ACL 侧保留 0/3/7），是**插件 Python 绑定缺失**（C++ 已有 `getStreamFromExternal`） | `prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L46–L62** |
| ⚠️ 机器级阻塞（**与本层无关**） | 共享 `Qwen3-Embedding-0.6B/model.safetensors` 被**存储层静默损坏**（2 张量 34396 个非有限值；`md127` RAID5 降级）⇒ 用 P800 位级原件在 scratch 自证修复（**只搬 2.33 MiB**）；**共享资产未改动** | `910C/docs/ASCEND_910C_R10_RERUN_20261008.md` **L89–L177** |
| 三芯片职责验收（09-22 傍晚 · 历史批次） | ✅ 10 项全绿 | `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md` |

→ 详见 `910C/README.md`

### 2.3 P800 实例（第二个接入实例，✅ **阶段 0–5 完成** + 镜像等价性已验证，09-22 验收 10 项全绿）

| 项 | 状态 | 证据 / 详见 |
|---|---|---|
| 阶段 0–5 全部完成（09-14 → 09-20） | ✅ 环境与五域基线 → 单卡接入 → 训练腿（loss 15.4488→11.1481 · **3533.5 tok/s**）→ 推理腿前向 **13/13**（53.12 句/s）→ 服务化 **10/10**（30.70 句/s）→ 错误闭环 **5/0/0** → 收敛三件套 | `P800/README.md` §0 · `P800/docs/KUNLUN_P800_ADAPT_PLAN_20260914.md` · `P800/docs/PROGRESS_REPORT_20260914.md` |
| ⭐ **职责审计（78 项口径 · 10-08）** | ✅ `DUTY_RESPONSE_PASS` **67 / 0 / 11**（SKIP 均如实不具备）· 非空转 **31 / 8 / 0** | `prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92** |
| ⭐ **流优先级** | ⚪ 本层**声明** `stream_priority_control`，但区间**退化为单点** `(0,0)` ⇒ **设置无区分**；「调度效果」如实 `NOT_APPLICABLE` | `prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54** · `prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L63–L78** |
| ⭐ **「不适用」自证审计（10-08）** | ✅ 成立：**逐卡 7/8**（跳过他人作业在用的卡）全 `(0,0)` · driver(`cu*`) 与 runtime(`cuda*`) **两层读数一致** · 全量 `nm -D` 无原生 XPU 流 API | `prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L63–L78** |
| 多流 Stream 16 项基线 | ✅ **14 通过 / 1 如实不支持（S-12 流优先级，上游缺陷）/ 1 不适用**；探针 **8/8**（与 910C 逐项一致） | `P800/docs/KUNLUN_P800_STREAM_BASELINE_16_20260920.md` |
| 官方 `-base` 镜像等价性（09-20） | ✅ 全部结论复现 ⇒ **缺陷与镜像无关**（KL3 挂死 A 组 3/3 一致重现） | `P800/docs/KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md` |
| 已知厂商缺陷（2 条 · 归厂商侧） | ⚠️ ① KL3 事件同步概率性挂死（≈89%，**不影响单进程设备上下文路径**）② **物理卡 1 计算通路故障**（控制面正常 ⇒「能查到卡」≠「能用卡」） | `P800/README.md` §3.1 / §3.2 · `P800/docs/KUNLUN_P800_CARD1_HANG_ISSUE_20260923.md` |
| 框架缺陷（第 4 例） | ✅ 已修在框架层：错误对象**跨模块类不相等** → `disposition` 取 `KeyError` | `P800/docs/KUNLUN_P800_STAGE34_VERIFY_20260920.md` §3 |

→ 详见 `P800/README.md`

### 2.4 关于"基于统一原型的训推复跑"（易混淆点，务必看清）

两条腿**都是基于 `prototype/` 的统一原型**（经 `runtime.use(...)` 接入设备）：

| 腿 | 脚本 | 接入方式 | 形态 |
|---|---|---|---|
| 训练腿 | `prototype/runtime/proto/proto_train_leg.py` | **后端无关化**（`DC_BACKEND` 环境变量驱动） | 910C → `use("ascend")`；P800 → `use("kunlun")`；MLU590 → `use("cambricon")` |
| 推理腿（前向） | `prototype/runtime/proto/proto_infer_leg.py` | **后端无关化 V2**（`DC_BACKEND` 驱动；设备串取 `runtime.current().device_type`，同步走 `runtime.synchronize()`） | 单卡 `transformers` 前向 + 真实异常注入（910C / P800 同一份脚本） |
| 推理腿（服务化） | `prototype/runtime/proto/proto_infer_serve.py` | 同上 | vLLM OpenAI 兼容服务（910C 108 句/s / P800 30.70 句/s，两实例均 10/10） |
| 错误闭环 | `prototype/runtime/proto/proto_error_recovery_loop.py` | 同上（补 `DC_BACKEND`） | 910C 两后端各自跑通；P800 两设置对照均跑通 |

但**不是**把历史那两套训推用统一原型重跑了一遍：

- `910C/distributed_training/` 的双卡 DDP（Qwen2.5-1.5B，loss 1.95）是**旧代码路径**
  ——直接 `import torch_npu` + `torch.distributed`，`runtime.use` 出现 **0 次**；
- `910C/distributed_inference/` 的 vLLM + TP（Qwen3-4B）同样是旧路径。

**如实标注的缺口**：历史模型（Qwen2.5-1.5B / Qwen3-4B）尚未在统一原型上复跑。

---

### 2.5 MLU590 实例（第三个接入实例，✅ **接入完成 · 12 项判定全部通过** · 2026-09-28）：环境/镜像 → 后端落地 → conformance 13+6 → **多流 16 项** → 训练腿 6/6（3015.3 tok/s）→ **推理腿前向 13/13** → **服务化 `SERVE_STANDARD_PASS`** → 错误闭环 5/0/0

| 项 | 状态 | 结果 / 详见 |
|---|---|---|
| 阶段 0–8（接入全链路） | ✅ **完成** | 环境普查 → 镜像定档 → 后端落地（13 抽象）→ 离线自检 → **conformance 13/13 + 6/6** → 多流 16 项 → 训练腿 **6/6**（`cncl`）→ 推理腿两形态 → 错误闭环 → 收敛；逐阶段证据 `MLU590/README.md` §0 |
| ⭐ **m1 补齐轮（10-08）· 20 项全绿** | ✅ 离线 **97/0/0** · 冒烟 46/0 · **职责 62/0/16** · 非空转 26/13/0 · 训练腿 6/6 · 推理腿 13/13 · 服务化 PASS + `SERVE_LEG_PASS` 10/10 | `MLU590/docs/CAMBRICON_MLU_M1_RERUN_20261008.md` **L10–L46** |
| ⭐ **流优先级（三家唯一能出「效果」结论的实例）** | ✅ 区间 `(0,-3)` **非单点** · **走厂商 CNRT C API**（与 kunlun 同构）⇒ 经统一面**真行使**契约 §1.10 **L5**；「调度效果」**分场景**：同时就绪 ⇒ 无实质效果 / 排队争用 ⇒ 有实质效果 | `prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54 / L94–L122** · `prototype/docs/STREAM_PRIORITY_READBACK_FIX_20261008.md` **L59–L100** |
| 推理腿（前向 + 服务化） | ✅ **13/13**（+1 如实跳过）：dim 1024 · **41.08 句/s** · p50 72.89 ms · 区分度 0.6391；服务化 **`SERVE_STANDARD_PASS`**（150 s 就绪，⚠️ **须用 vLLM 应用镜像容器**） | `MLU590/docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md` |
| 训练腿 2 卡 | ✅ **6/6**：loss 15.4498→11.1479 · **3015.3 tok/s** · `dist=cncl`（⚠️ 走 **MLU_LINK 片间互联**，非 RDMA —— 已如实标注） | `MLU590/README.md` §0 阶段表 |
| 镜像档位 | ✅ 定档 `flagos-runtime-cambricon-neuware4.4.3:2.2.0`（**选档第一判据 = 宿主驱动 v6.2.29**）；4.7.2 档列为**上报预案** | `MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` **L10–L109** |
| ⚠️ 一条原预期被实测否定 | CNRT 抛的是**错误名**而非数字码 ⇒ **无可建码表** ⇒ `error_map` 如实不声明（分级由 `message_hint` 覆盖） | 同上文档 · `MLU590/README.md` §0 |


**两处实测要点**（详见 `MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md`）：
1. **建不了 `/srv/hliu553`** —— `/srv` 属主 `root:root 755`，实测 `mkdir: Permission denied`；虽在 `sudo` 组但 sudo 需密码 ⇒ 需 root 开通；
2. **不在 `docker` 组** —— 而我们的验证流程全部在带卡容器内 ⇒ 与第 1 条并列的硬阻塞。

> **完整方案与真机执行手册**：[`MLU590/docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md`](MLU590/docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md)（含 A1–A10 真机步骤与确切命令、验收清单 13 项当前状态、风险与应对）。
>
> **⭐ 一条方法学教训（已写进指南）**：**选镜像档位的第一判据是宿主驱动，不是 Python 包版本。**
> 本次先按 `torch-mlu 1.33.1` 锚档，核查后发现那是 4.7.2 档、而它要求宿主驱动 6.5.48（我们只有 6.2.29）
> ⇒ 已按驱动修正为 4.4.3 档。判据与三实例对照见 `prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md`。

→ 详见 `MLU590/README.md`

### 2.6 PPU 实例（第四个接入实例，🟢 **接入自检 + 两条腿已达三家同口径** · 2026-10-10）：后端落地 → conformance 13+6 → 标准步骤 6 项 → **步 0 模型完整性** → **训练腿 50 步 6/6** → **推理腿 13/13**；⚠️ **服务化/错误闭环/非空转未做**；✅ **职责审计 72/0/6（`L1` 根因已定位并修复）**；⚠️ 退出期段错误未收口

| 项 | 状态 | 结果 / 详见 |
|---|---|---|
| 形态 / 环境 / 后端 | ✅ **完成** | 形态 **路径 B（复用 `torch.cuda`）实测钉死** · 后端名 **`nccl`**（⚠️ 库才叫 PCCL）· 16×`PPU-ZW810E`（96 GB/卡）· 基座与同机「工作负载」容器**对齐**（不自定）；报告见下行 |
| 自检（离线 / 对称 / smoke / conformance） | ✅ 全绿 | 离线 **112/0/1** · 对称性 **7/0** · 真机 smoke **46/0** · conformance **13/13 + 6/6**；⬇️ 报告 A |
| ⭐ **步 0 模型完整性** | ✅ **PASS** | `Qwen/Qwen3-Embedding-0.6B`（HF revision `97b0c614…`，与 P800/MLU590 **同一份**）· 6 个关键文件 sha256 **与 P800 逐字一致** · CPU/PPU 前向均无 NaN；⬇️ 报告 B **§1–§2** |
| **训练腿（2 卡 · 50 步）** | ✅ **`TRAIN_LEG_PASS 6/6`** | loss **15.4498→11.1530** · **5167 tok/s**（两卡合计）· `dist=nccl`；⚠️ 首轮误用 20 步（判据强度低于三家）⇒ **自审后已按手册 50 步重跑** |
| **推理腿（单卡）** | ✅ **`INFER_LEG_PASS 13/13`** | dim 1024 · **135.77 句/s** · p50 **22.15 ms** · 区分度 **0.6391** |
| **流优先级** | ✅ 区间 **`(0,-3)` 非单点**（第 2 家多档） | C API 建流（flag=1）**请求 == 回读**；⬇️ 报告 A §3 |
| ⚠️ **退出期段错误** | ⚠️ **未收口 · 归属未定** | 两条腿**判据通过、结果 JSON 已落盘之后**在解释器退出期 SIGSEGV（训练腿两 rank `-11`；推理腿 `rc=139`）；已定位到与 `runtime.create_stream()` **强相关**（仅去掉该行即干净退出），**精确触发条件未收敛**（16 组对照）；⬇️ 报告 B **§4** |
| ⚠️ 两个容器级前置 | ✅ 已固化 | `-e NCCL_SOCKET_IFNAME=bond0` · `--add-host=ai-server:127.0.0.1`（后者缺则 **`torchrun` 静默挂 300 s 且无任何子进程输出**）；⬇️ 报告 B **§5** |

> 报告 A = `PPU/docs/PPU_BACKEND_ONBOARDING_20261010.md`（接入）｜
> 报告 B = `PPU/docs/PPU_MODEL_AND_TWO_LEGS_20261010.md`（模型 + 两条腿 + 段错误排查）
> → 详见 `PPU/README.md`

## 3. 快速入口

```bash
# 原型自检与 conformance（任一后端）
cd dev/device-context/prototype
python3 runtime/smoke_runtime.py                          # 冒烟自检（不传 --backend 则自动挑选可用后端）
python3 runtime/conformance/runner.py --backend ascend    # 昇腾后端
python3 runtime/conformance/runner.py --backend cambricon  # 寒武纪后端（MLU590）
python3 runtime/conformance/runner.py --backend kunlun    # 昆仑芯后端（P800）

# 两条腿（后端由环境变量决定）
torchrun --nproc_per_node=2 runtime/proto/proto_train_leg.py     # 训练腿
python3 runtime/proto/proto_infer_leg.py                         # 推理腿
```

**P800 侧**：探针与原始证据见 `P800/probes/`；**可复现命令统一入口**见
`prototype/docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md` §6（含镜像补齐、conformance、两条腿、KL3 对照）；
根因取证见 `P800/docs/KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md`。

> ⚠️ **若在官方 `-base` 镜像上跑**，开工前须先补齐 `triton`（否则推理腿服务化报
> `Failed to infer device type`）：`pip install flagtree===0.7.0rc3+xpu3.6 --index-url=https://resource.flagos.net/repository/flagos-pypi-hosted/simple`，
> 并设 `PYTHONPATH=/env/FlagGems/src`（两条均为实测硬前置，详见
> `prototype/docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md` §2.1.1 C）。

---

## 4. 关键约束（务必先读）

**`dev/stack.lock.910c.v2.yaml` 置顶规则：带卡容器并发上限 3（910C）。**
⚠️ **实测更正（2026-09-28）**：**不要把 3 当可用阈值** —— 09-22（我方 2 个带卡容器共存）与
09-28（他人 2 + 我方 1 = 3 个）两次都失败，且 09-28 用「只挂 1 张无人占用卡的临时容器」验证**同样失败**
⇒ **实践中按「同一时刻只留 1 个带卡容器」安排**（详见 `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md` §4.3）。
（P800 无此限制，但**用卡前必须 `xpu-smi` 挑「连续且空闲」的卡并记录用卡**）

- 名额用尽时 `acl.init()` 返回 **500000**（`ACL_ERROR_INTERNAL_ERROR`），表现为 `device_count=0`、设备"消失"
  （`acl.init rc=0` 但 `get_device_count=(0,0)` 也属此列）
- 出现该现象**第一时间核查并发容器数**，不要先怀疑镜像/驱动/代码
- 他人容器占用时：**先协调**，跑完立即用 `docker start` **原样恢复**（容器未删、配置不变）

| 腿 | 芯片 | 镜像 | 设备后端 | 额外约束 |
|---|---|---|---|---|
| 训练 | 910C | `flagrt/ascend-operator-runtime-comm:0.1.3` | **`npu`（torch_npu）** | 训练腿 2026-09-22 已统一 torch_npu；用容器内只装 torch_npu 的解释器即物理隔离（**镜像未切换**） |
| 推理 | 910C | `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3` | **npu（torch_npu）** | 选卡 `ASCEND_RT_VISIBLE_DEVICES` |
| 训练／推理 | P800 | 现用 `flagtree-xpu3.6-py310-torch2.9.0-flaggems-main-dev:202608`；**建议入锁** `harbor.baai.ac.cn/flagtree/flagtree-xpu3.6-py310-torch2.9.0-ubuntu22.04:202608-base`（等价性已验证） | **kunlun（`torch.cuda`）** | 选卡 `CUDA_VISIBLE_DEVICES`；需 `FLAGCX_ADAPTOR=klx` + 显式 `import flagcx`；**不设 `XPU_EVENT_KL3_ENABLE`**（见 `P800/README.md` §3）；若用官方 `-base` 须先补 `triton` 并设 `PYTHONPATH=/env/FlagGems/src` |

---

## 5. 近期动作

> 完整动作日志（**60 项**，2026-08 → 2026-10）已移入
> [`prototype/docs/BOARD_ACTION_LOG_202608_202610.md`](prototype/docs/BOARD_ACTION_LOG_202608_202610.md)；
> 本表只留**近 10 轮**（每轮一行 + 报告与行号）。历史轮次的口径以当时报告为准。

| 轮次 | 动作 | 状态 | 详见（报告 + 行号） |
|---|---|---|---|
| **10-10 · 17** | **PPU 补齐模型与两条腿**：模型经 hf-mirror 取得并与 P800 副本**逐字 sha256 对照**；**训练腿 6/6 · 推理腿 13/13**；⚠️ 两条腿**退出期段错误**（未收口、归属未定）。同轮修掉两处**跨实例共性缺陷**：① `_DIST_BT_DEFAULT` 缺 `ppu` 项 ⇒ 不给 `DC_DIST_BT` 会**静默回落 `gloo`**（纯 CPU 集合通信，训练照跑 loss 照降），已补并改为「表中无项即拒绝」；② `--network host` 下主机名不解析 ⇒ **`torchrun` 静默挂 300 s 且无任何子进程输出**，已固化 `--add-host` | ✅ / ⚠️ | `PPU/docs/PPU_MODEL_AND_TWO_LEGS_20261010.md` **§1–§6** · `prototype/runtime/proto/proto_train_leg.py` **L62–L113** |
| **10-08 · 16** | 回答「910C / P800 的『流优先级不适用』**属不属于本层职责、能不能修**」→ 两家**都不属、都不可修**；**顺带核出并修掉我方一处口径分叉**（契约 §1.9 表 P800 行与代码/看板/台账矛盾，自第九轮起漏改） | ✅ | `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` **§1.9** · `prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md` **第 36 条** |
| **10-08 · 13 续** | **D2 调度效果对照实验收尾** + 910C 补跑 ⇒ **三家现役实例全覆盖**；MLU590 出**分场景结论**，910C/P800 如实 `NOT_APPLICABLE`（**原因不同**） | ✅ | `prototype/docs/STREAM_PRIORITY_SCHED_EFFECT_20261008.md` **L21–L54 / L123–L150** |
| **10-08 · 13** | **「不适用」自证审计**：按**六问**重新取证 ⇒ 两者结论**成立**，910C 根因**升级为 Python 绑定缺失** | ✅ | `prototype/docs/STREAM_PRIORITY_NOT_APPLICABLE_AUDIT_20261008.md` **L16–L112** |
| **10-08 · 12** | **流优先级「回读口径」修复**：更正「三家都行使不了 L5」；MLU590 回读原为**空转判据**（读构造参数回显）⇒ 改走**厂商 C API**；修 **id 复用越权销毁** | ✅ | `prototype/docs/STREAM_PRIORITY_READBACK_FIX_20261008.md` **L10–L143** |
| **10-08 · 11 (m1)** | **MLU590 补齐轮**：一次窗口 **20 项全绿**（含步 0 **模型完整性**）；台账 **D1 收尾**；同轮**暴露并修 5 处缺陷** | ✅ | `MLU590/docs/CAMBRICON_MLU_M1_RERUN_20261008.md` **L10–L46 / L110–L199** |
| **10-08 · 10 (r10)** | **910C 第 10 轮复跑**（台账 **D4 收尾**）：**20 项全绿**；过程遇**共享模型被存储层静默损坏**（非本层，已自证修复） | ✅ | `910C/docs/ASCEND_910C_R10_RERUN_20261008.md` **L10–L49 / L89–L177** |
| **10-08 · E1** | **职责审计口径与契约齐平（39 → 78 项）**，三实例如实结果 73/0/5 · 67/0/11 · 62/0/16；顺带补 2 个**统一面出口** | ✅ | `prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md` **L42–L92** |
| **09-30 · 9** | **流优先级统一 API 落地**（`create_stream(priority=)` + 回读校验 + 三把钥匙）；P800 由「不声明」改为**真声明** | ✅ | `prototype/docs/STREAM_PRIORITY_API_20260930.md` **L11–L106 / L265–L348** |
| **09-30** | **B2 收尾**（配额真实上限 **1979**、优先级两层障碍）· **C2 宿主副本对齐** | ✅ | `prototype/docs/WORKPACKAGE_D_STREAM_PRIORITY_QUOTA_20260930.md` · `prototype/docs/ENV_AND_HOSTCOPY_CLOSURE_C2_C3_20260930.md` |
| **09-29 · 5–8** | 上下文语义专项 · **A2 多卡 `real` 压测** · **补公开入口**（`set_device_state` / `handle_error`）· **B3 修订建议逐条复核** | ✅ | `prototype/docs/PUBLIC_ENTRYPOINTS_RECOVERY_20260929.md` · `910C/docs/ASCEND_910C_A2_RECOVER_STRESS_20260929.md` · `prototype/docs/INTERFACE_CONTRACT_REVISION_STATUS_REVIEW_20260929.md` |

---

## 6. 文档总目录（**下游按此取用**）

> **先看这张表判断该读哪份**：文档分三层效力——**规范/效力文件**（你**必须遵守**，我们要走变更流程）、
> **操作手册/标准**（照做即可）、**参考/实测记录**（描述我们当时怎么做的，不含承诺）。
> 芯片专属结论一律放 `910C/`、`P800/` 目录下，**原型与规范侧（`prototype/`）芯片无关**。

### 6.0 四条阅读路径（按你的角色选）

| 你的目的 | 读这几份 | 顺序 |
|---|---|---|
| **我要按统一接口对接**（显存/分布式/监控/精度/算子/调度方向） | ① `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md` → ② `prototype/docs/event_semantics_contract.md` → ③ 修订建议（了解将变项） | ①②③ |
| **我要起推理服务** | ① `prototype/docs/SERVICE_STARTUP_STANDARD_20260920.md` → ② 脚本 `prototype/scripts/serve_standard.sh` | ①② |
| **我要接入新芯片** | ① `prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md` → ② 参考实例（`P800/docs/KUNLUN_P800_ADAPT_PLAN_20260914.md`，已完成的第 2 家）→ ③ 第 0 步环境报告范例（`MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md`，进行中的第 3 家） | ①②③ |
| **我要复核我们说过的话** | ① `prototype/docs/VERIFICATION_MANIFEST_20260920.md`（**10 条**命令） → ② 原始证据 `910C/probes/`、`P800/probes/`、`MLU590/probes/` | ①② |

### 6.1 规范 / 效力文件（**下游必须遵守**；变更需走流程）

| 文档 | 回答什么 | 读者 | 变更流程 |
|---|---|---|---|
| `prototype/docs/INTERFACE_CONTRACT_DC_20260908.md`（427 行） | **我承诺什么接口语义**：统一 API 面（`use` / 设备 / 流 / 事件 / 同步含超时 / 错误翻译 / 状态与恢复）、**Backend 插件接入规范（13 个抽象方法）**、两条纪律 | 运行时层各子方向 + 上层算子/编译层对接人 | 更新文档 → **知会全部下游** → conformance 回归 |
| `prototype/docs/event_semantics_contract.md`（57 行） | **事件语义契约 E1–E4**（昇腾实测驱动的 v2 修订）：事件/流的同步与依赖语义边界 | 同上 | 同上 |
| `prototype/docs/SERVICE_STARTUP_STANDARD_20260920.md`（267 行） | **你们必须怎么起服务**：唯一入口 `serve_standard.sh`、参数表、**六条硬纪律**、三个已知行为、与自建脚本的关系 | 所有需要在国产芯片上起服务的方向与验收方 | 改脚本 + 改文档 → **两实例各跑一次** → 知会下游 |
| `prototype/docs/INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`（549 行） | **接口约定修订建议 9 条**（v1.1；P800 是现行约定的首次非昇腾检验）：`device_type`/`vendor` 分离 · `device_state` 入契约 · `.native` 逃生舱约束 + `record_stream` 能力位 · **错误对象跨模块类归一** · 有界同步降级语义 · `known_issues()` 入契约（**v1.1 另增第 7–9 条**）。每条含 ①现状 ②实测依据（带源码行号）③建议条文 ④兼容性 | 接口约定评审人、全部下游 | **尚未生效**——待裁定后并入接口约定，届时按 6.1 第 1 行走流程 |

### 6.2 操作手册 / 标准（照做即可）

| 文档 | 回答什么 |
|---|---|
| `prototype/docs/NEW_CHIP_ONBOARDING_MANUAL_20260920.md`（449 行） | **新芯片怎么接进来**：第 0 步环境风险前置 7 项 → 第 1 步厂商栈判别 4 条路径 → 第 2 步镜像就绪 5 条判据 → 第 3 步 13 个抽象方法清单 → 第 4 步 conformance（§4.4 含**离线契约自检**，无设备即可跑）→ 第 5 步两条腿 → 第 6 步错误闭环 → **可勾选验收清单 13 项** + 跨芯片坑 13 条 + 厂商缺陷上报模板 |
| `prototype/docs/VERIFICATION_MANIFEST_20260920.md`（179 行） | **怎么复核**：**10 条**「声明 → 命令 → 判据」最小复现表 · **三实例**证据索引（含"当前结论 = 哪一份"）· 缺口 G1–G8 · **证据命名规范** |
| `prototype/scripts/serve_standard.sh`（412 行） | **服务启动唯一入口**：`DC_BACKEND` 切芯片、`SERVE_FORM=generate|embed` 切形态（留空＝各后端现状）、`SMOKE_TIMEOUT` 控冒烟超时（默认 180 s）、`RELEASE_WAIT` 控停机后释放回落等待（默认 30 s，v1.3）；流程 = 服务入口就绪 → 设环境 → 清残留 → 用卡快照 → 启动 → 就绪轮询 → 功能冒烟 → 停机复查（轮询到回落稳定）；verdict = `ready=1 且 smoke=1` |
| `prototype/scripts/preflight_env.sh`（201 行） | **环境普查一键脚本**（= 接入手册 §1「环境风险前置 7 项」的可执行版）：只读、不装东西；含 docker 数据目录真实挂载点、torch 侧降级查询、网络源可达性、拓扑；缺项如实标注「未取得」，输出可直接作为环境报告 |
| `prototype/docs/RUNTIME_PROTOTYPE_DESIGN_20260904.md`（314 行） | **原型怎么设计的**：五域划分、13 个抽象方法的来由、目录结构、验证方式（v0.1） |
| `prototype/docs/RUNTIME_DC_STREAM_PLAN_20260907.md`（177 行） | **本层职责边界与方法**：设备抽象 / 多流 Stream / 错误码翻译 / 状态恢复的职责划分，与上下游分工，**多流验收基线 16 项的出处** |
| `prototype/docs/RUNTIME_LAYER_MONTHLY_PLAN_20260908.md`（150 行） | **月度里程碑与交付物口径**（运行时层 9 月聚焦版） |

### 6.3 参考 / 实测记录（描述"我们当时怎么跑的"，不含承诺）

| 文档 | 回答什么 |
|---|---|
| `prototype/docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md`（335 行） | **两实例配置与依据**：镜像 / 模型 / 训推框架 / 参数逐项对照 + **为什么这么定**（依据写成可独立阅读的实测事实）+ §4 结果可比性说明 + §5 坑清单 + §6 可复现命令 + §7 未覆盖项。**主线汇报材料** |
| `prototype/docs/BACKEND_SYMMETRY_AUDIT_20260922.md` | ⭐ **跨后端对称性审计台账**：**第 6/7/8 条跨后端缺陷**（诚实降级三字段不一致 ⇒ 假"确定分级"；某后端声明 `device_state` 却无实现；`info()` 能力键名漂移）+ **2 处证据污染**（硬编码昇腾码 `507046`、L4 注入描述对无码表后端是错的）；每条给"现象 / 机制 / 归属 / 处置 / **防回归判据** / 各实例验证状态"，并含**判据非空转验证**与工具侧改进 |
| `prototype/docs/IMAGE_SELECTION_GUIDE_20260920.md`（203 行） | 镜像怎么选：需求画像（5 条可执行判据）· 来源优先级（**2026-09-22 起 FlagOS 官方 `flagos-base`/`runtime`/`app` 体系列为最高优先级**）· **入档/入锁两道门槛**（含"谁定"）· 实操五步 |
| `prototype/docs/IMAGE_REQUIREMENT_SPEC_20260920.md`（139 行） | **向上游要什么**：可查验的官方文档清单 · 我们镜像与官方推荐的差异 · 硬性 H1–H6 / 期望 E1–E5 / 可协商 N1–N3 三级需求 · **请总组裁定的三件事** |
| `prototype/docs/IMAGE_LINEAGE_ALIGNMENT_20260922.md`（新建） | **镜像血统对齐核查（三家对照）**：类脑（x-benchmark）只覆盖 FlagTree 算子线（4 家、**无寒武纪**）· **FlagOS 官方镜像体系**（`flagos-base`/`runtime`/`app`，14 后端含寒武纪，匿名可拉实测）· **宿主驱动是选档第一判据**（寒武纪 6.2.29 → 只能 4.4.3）· 昆仑芯两条血统（FlagTree xpu3.6 vs FlagOS XRE 5.37.1）· **官方对 KL3 缺陷的印证** |
| `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260928.md`（**新建**） | ⭐ **三芯片职责验收与发布结论（09-28 版）**：职责定义（五域 13 抽象 + 两条硬纪律 + **覆盖映射**）· **三实例 12 项判定矩阵** · 本轮 1 处修复（冒烟超时假失败）+ 1 项解法（服务化改应用镜像）· 发布结论与遗留 |
| `prototype/docs/PROTOTYPE_ACCEPTANCE_3CHIP_20260922.md`（**上一版**） | 三芯片职责验收（09-22）：当时 MLU590 主机不可达 ⇒ 未纳入；**记录保留不覆盖** |
| `prototype/docs/ROUTE_B_ARCHIVED_20260922.md`（129 行） | **路线 B（torch_fl）退出归档**：删了什么 / 保留什么 / 残留全量清单 / 复跑清单 |
| `prototype/docs/DESIGN_DIST_COMM_20260908.md`（69 行） | 2 卡分布式微调的通信路线思考备忘（**状态：思考结论，尚未实跑**）；与分布式方向的接口约定**待回复** |
| `prototype/README.md`（182 行） | 原型分支看板：统一 API 面、目录结构、**三实例**验证状态、文档索引 |
| `prototype/RELEASE_NOTES_v0.3.0.md`（218 行） | ⭐ **组件 v0.3.0 发布说明（当前版）**：契约 §1.6–§1.10 五个 sub-part · 统一面出口 28 → 46 · **8 处行为修正**（含「误触发设备级重建」「永久参数错误反复 replay」）· 调度效果**分场景结论** · 已知限制 11 条 · 升级建议 |
| `prototype/RELEASE_NOTES_v0.2.0.md`（222 行） | **组件 v0.2.0 发布说明**（**三芯片统一原型版**；发布页 https://github.com/FlagRT/runtime-team/releases/tag/runtime-v0.2.0）：kunlun 后端 · 4 个框架修复 · 脚本后端无关化 · 验证结果 · **纪律 3 条** · 已知限制 9 条 |
| `prototype/RELEASE_NOTES_v0.1.0.md`（109 行） | 组件 v0.1.0 发布说明（初版，910C 单实例） |
| `prototype/probes/probe_stream_semantics_full.py` | **多流 16 项基线探针（后端无关 V2）**：覆盖 S-1/S-2 补强 + S-8~S-13，设备 API 前缀由统一运行时给出，同一份脚本跨芯片复用（`DC_BACKEND` / `DC_TAG`） |
| `prototype/docs/MERGE_PREP_DEV10_20261008.md`（137 行） | ⭐ **合入 `dev-1.0` 的准备材料**：子树同步方式（基 `dev-1.0` tip，只落 `dev/device-context/` + `summary/`）· **合入把关四道逐条读数**（可快进 / 子树 tree 逐字相同 / 禁用词 0 / 基座草稿未带入）· 可直接使用的 **PR 标题与正文** · 合入后收尾 · 未做项 |
| `../../summary/DEVICE_ABSTRACTION_ROUTE_AB_SUMMARY_20260922.md` | **分支级总结**（跨目录，在仓库根 `summary/`）：路线 A/B 选择依据 + 路线 A 设计方案 + **三实例**实现进度与下一步 |

### 6.4 芯片专属文档（**结论不迁移**，新芯片按手册新建）

**6.4.1 `910C/docs/`（昇腾，第一实例）**

| 文档 | 一句话说明 |
|---|---|
| `ACL_ERROR_MAP_20260901.md`（173 行） | ACL 错误码映射表建设记录（D10）：**108 条码 → L1–L4 分级依据**。⚠️ **不迁移**给新芯片 |
| `ASCEND_910C_DC_STREAM_MAPPING_20260902.md`（809 行） | 910C 设备上下文 × Stream 双侧全景：职责映射与实测；**含 S-1～S-16 编号基线**（多流验收基线的口径来源） |
| `DC_STAGE_SUMMARY_20260909.md`（171 行） | 设备上下文阶段性总结：两条腿证据并入 |
| `ERROR_RECOVERY_LOOP_20260909.md`（150 行） | 错误注入 → 恢复闭环验证记录（含一处归因核查被推翻的记录） |
| `DIAG_TRAIN_IMAGE_NPU_20260908.md`（64 行） | 训练腿锁定镜像 NPU 初始化失败排查记录 |
| `DEVICE_CONTEXT_PLAN_20260827.md`（117 行） | 设备执行上下文方案定稿（8 月版） |
| `PROGRESS_20260822.md`（84 行） | 8-22 阶段进度快照 |
| `PR_DEV_1_0_20260902.md`（217 行） | PR #11 描述（同构 910C 职责验收，157 文件） |
| `910C-env-issue-report.md`（59 行） | 容器内 `aclInit` 返 500000 问题记录（**根因 = DrvMng 容器上限 3**，已解决） |
| `O3_getlasterror_fix.md`（179 行） | FlagCX O3 缺陷（`flagcxGetLastError` 存根完善）设计与实现 |
| `O4_socket_seq_guard.md`（151 行） | FlagCX O4 缺陷（socket 协议无 tag 匹配）暴露点分析与加固 |
| `OFFICIAL_RUNTIME_COUNTERPART_20260922.md` | **FlagOS 官方 910C 对应镜像对照**（2026-09-22 新增）：`flagos-runtime-ascend-cann9.0.0-910c:2.2.0` 与我们锁定栈**逐项一致**（CANN 9.0.0 / py3.11 / torch 2.10.0 / triton 3.5.0 / **flagtree 0.7.0rc2+ascend3.5**）；**设备后端为 `npu`（Route A）** —— 与**我们已统一的口径一致**（2026-09-22 起训练腿已走 torch_npu，路线 B 的权宜例外取消）⇒ 差异只剩镜像血统；⚠️ **官方 runtime 不含 FlagCX** ⇒ 若切它需另解 FlagCX。**仅登记，未切换**（同源登记在 `dev/stack.lock.910c.yaml` 的 `candidates:`） |

**6.4.2 `910C/distributed_inference/docs/`（910C 推理既有工作）**

| 文档 | 一句话说明 |
|---|---|
| `DEVICE_CONTEXT_INFERENCE_MAPPING_20260831.md`（398 行） | 设备上下文 × 分布式推理 · 工作与验证映射（**验收依据**） |
| `DEVICE_CONTEXT_INFERENCE_PLAN_20260831.md`（148 行） | 推理 × 设备上下文/Stream 实现方案 |
| `INFERENCE_P0_P1_RUN_20260831.md`（170 行） | 推理 P0/P1 首轮执行记录 |
| `INFERENCE_P3_SERVE_STATE_ERROR_20260901.md`（191 行） | P3 服务化 × 设备状态/错误恢复执行记录（A8/A9/A10，验收闭环） |
| `INFERENCE_QWEN3_TP_COMPARE_20260901.md`（104 行） | Qwen3-4B TP 数值等价性验证（⚠️ 须 greedy，temperature=1 必然发散） |

**6.4.3 `910C/distributed_training/docs/`（910C 训练既有工作）**

| 文档 | 一句话说明 |
|---|---|
| `DEVICE_CONTEXT_TRAINING_MAPPING_20260831.md`（90 行） | 设备上下文 × 分布式训练 · 映射与验收评估 |
| `FLAGCX_CORE_DEFECT_FIXES_20260826.md`（293 行） | FlagCX 核心库原生 allreduce 缺陷修复全过程（死锁 P2 + 数据错乱 P6 + 显存 OOM P7） |
| `netcc_chunk_race_investigation.md`（248 行） | net.cc chunk 流水线偶发数据错：源码级调研与修复方案 |
| `flagcx_ascend_aline_validation_20260824.md`（63 行） | 昇腾 A 线验证报告 |
| `4090_training_report.md`（58 行） | 4090 两卡 1.5B 训练报告（FlagCX NVIDIA 适配） |

**6.4.4 `P800/docs/`（昆仑芯，第二实例）**

| 文档 | 一句话说明 |
|---|---|
| `KUNLUN_P800_ENV_REPORT_20260914.md`（297 行） | 基础环境汇总（只读探测：8 卡 / 1.5 TiB / 384 线程 / 200 Gb RoCE / 拓扑分组） |
| `KUNLUN_P800_BASELINE_PROBE_20260914.md`（178 行） | **五域基线实测**与缺失项归属判定 |
| `KUNLUN_P800_ADAPT_PLAN_20260914.md`（453 行） | **接入工作方案**：定位与交付边界、接入路线、验收 6 条、阶段计划、风控；**§7.5 = 接入过程暴露的 3 个「只有非昇腾实例才暴露」的框架缺陷** |
| `KUNLUN_P800_ROOT_CAUSE_VERIFY_20260914.md`（240 行） | 集合通信挂死结论核对与**责任层判定**（两要素条件、函数级定位到厂商 `.so`） |
| `KUNLUN_P800_STAGE34_VERIFY_20260920.md`（240 行） | **阶段 3/4 验证**：推理腿 13/13 与 910C 同构对照 · 错误闭环两设置对照（逐字节一致）· **§3 第 4 例框架缺陷的根因与修复** |
| `KUNLUN_P800_BASE_IMAGE_EQUIVALENCE_20260920.md`（288 行） | **官方 `-base` 镜像等价性验证**：§0 镜像速查（tag/digest/大小/获取与补齐三步）· 全部结论逐项复现 · **缺陷与镜像无关**（排除"是我们镜像的问题"）· 入锁建议 |
| `KUNLUN_P800_STREAM_BASELINE_16_20260920.md`（219 行） | **多流 16 项逐项比对**：14 通过 / 1 如实声明不支持 / 1 不适用；S-7 图捕获首测 5/5；含一处自我纠错与证据形态差异说明 |
| `PROGRESS_REPORT_20260914.md`（859 行） | 全量进度报告（910C 回顾 + P800 主体 + **待办按"谁来做"四分类** + 证据索引） |

**6.4.5 `MLU590/docs/`（寒武纪，第三实例）**

| 文档 | 回答什么 |
|---|---|
| `MLU590/docs/CAMBRICON_MLU_ENV_REPORT_20260922.md` | **环境报告（第 0 步）**：两机并列明细 · docker 数据盘归属证据链 · MLU 栈版本组合 · root 权限开通需求 · 探测边界。⚠️ **其中「torch-mlu 1.33.1 + torch 2.11.0」属 `neuware4.7.2` 档（需驱动 6.5.48）**，已被本方向定档 `neuware4.4.3`（py3.10 / torch 2.7.1 / torch-mlu 1.29.2）取代，见 `CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` §0.1 |
| `MLU590/docs/CAMBRICON_MLU_INFER_LEG_VERIFY_20260928.md` | ⭐ **推理腿（前向 + 服务化）验收报告（09-28）**：12 项判定全通过 · 两条要点（**运行时镜像不含 vLLM ⇒ 用官方应用镜像**；**冒烟超时硬编码 60 s ⇒ 假失败**，含 PRE_FIX 原样留档）· 三实例对照与差异解释 · 边界与未覆盖 · 复现命令 |
| `MLU590/docs/CAMBRICON_MLU_ADAPT_PLAN_20260922.md` | ⭐ **接入方案 + 真机执行手册**：进度表（第 0/0b/1/2/3 步状态）· 厂商栈判别（预期 **路径 C：PrivateUse1 / `mlu`**）· 已完成的代码层动作与**能力声明逐条理由**（为什么 `error_map`/`recovery_real`/`graph_capture`/`stream_priority` **如实不声明**）· 本地验证（**离线自检 35/0**）· **A1–A10 真机执行序列（含确切命令与出处）** · 验收清单 13 项当前状态 · 风险与应对 · 职责边界 |
| `MLU590/docs/CAMBRICON_MLU_IMAGE_CHANNEL_20260922.md` | **镜像获取渠道调研 + 当日更正**：§0 更正段（**FlagOS 官方 BAAI Harbor 已有寒武纪三代镜像、实测可匿名拉取**；**档位由宿主驱动决定**：6.2.29 → `neuware4.4.3`，`neuware4.7.2` 需 6.5.48）· FlagTree 无寒武纪手册 · 官方渠道与三私仓实测（均 401 需鉴权）· 申请清单 · **三家实例镜像获取路径对照** |

**6.4.6 `PPU/docs/`（平头哥，第四实例 · 已上机接入）**

| 文档 | 回答什么 |
|---|---|
| `PPU/docs/PPU_BACKEND_ONBOARDING_20261010.md`（115 行） | ⭐ **后端接入报告（2026-10-10）**：形态判定（**路径 B 实测**）· 环境与栈读数 · ⭐ **能力逐项实测表**（含两处如实不声明：`error_map` / `recovery_real`）· 三条实测发现（`eth0` 坑 / 后端名 `nccl` / E3 缺口）· **从既有工具挖出的 3 处缺陷** · 未完成项（**10-10 续报已标注状态**）。验证：离线自检 **112/0/1** · 对称性 **7/0** · 真机 smoke **46/0** · conformance **13/13 + 6/6** |
| `PPU/docs/PPU_ONBOARDING_PARITY_AUDIT_20261010.md`（约 150 行） | ⭐ **同口径自审报告（2026-10-10）**：以手册 §7 13 项 + 扩展步骤为口径，按**证据文件**（非文档声明）逐项比对四家；查出并处置 **3 类问题**（训练腿 20 步口径不齐→已按 50 步重跑 · 6 个芯片无关标准步骤缺失→已补齐全绿 · **职责审计 1 条 FAIL ⇒ 根因＝跨类型地址复用，已修复（类型复核，机制无关）**，另有 `PPU_L1_ROOTCAUSE_RECHECK_20261010.md`）；列明**仍未同口径**的 3 项（服务化 / 错误闭环 / 非空转） |
| `PPU/docs/PPU_MODEL_AND_TWO_LEGS_20261010.md`（198 行） | ⭐ **模型资产与两条腿（2026-10-10）**：模型三条取法实测比较（**官方站不通 · hf-mirror 25 s 下完**）· **步 0 完整性**（sha256 与 P800 逐字对照）· **训练腿 6/6 / 推理腿 13/13** 读数 · ⚠️ **§4 退出期段错误 16 组对照矩阵**（已排除项逐条列出）· **§5 容器级前置与修法**（主机名不解析 ⇒ torchrun 挂 300 s） |
| `PPU/docs/THEAD_PPU_FLAGOS_ADAPTATION_RESEARCH_20261009.md`（302 行） | ⭐ **接入前调研（含复核）**：生态对该芯片的适配全景（**组件层三件齐备 · 运行时层与基座层镜像为 0 · 档位三套并存**）· 接入形态判定（**间接证据指向「复用 CUDA 命名空间」**，⚠️ 直接判定**待上机**）· 与前三实例的对照 · 风险 **9 条** + 待确认 5 项 · ⭐ **§8 复核记录**（8 条逐条重新取证、含 1 条降级；并列出**仍属未验证的 4 项**）。⚠️ **未上机，不含任何设备侧结论**（形态一条已由上机实测确认） |

### 6.5 原始证据目录（复核用，勿只读结论）

| 目录 | 内容 | 约定 |
|---|---|---|
| `910C/probes/` | 第一实例原始证据：一致性判据 / 语义基线 / 错误闭环 / 推理腿 / 训练腿（含 2026-09-22 对称复跑 `recheck_*` 7 份）+ 统一脚本验证日志 | `.log` 需目录内 `.gitignore` 开 `!*.log` 例外 |
| `P800/probes/` | 第二实例原始证据：接入探针、两腿、服务化、错误闭环、镜像等价性对照（含 `recheck_*` 4 份） | 同上 |
| `MLU590/probes/` | 第三实例原始证据：16 项基线、两条腿、服务化、错误闭环、m1 补齐轮 | 同上（索引见 `MLU590/docs/EVIDENCE_INDEX_MLU590.md`） |
| **`PPU/probes/`** | 第四实例原始证据：（a）**接入轮** `onboard_20261010_out/`（离线自检 · 对称性 · 真机 smoke · conformance 13+6）+ 4 轮能力探测探针；（b）**模型与两条腿轮** `model_and_legs_20261010_out/`（步 0 校验 · 模型清单与 sha256 · 两腿结果 JSON 与日志 · ⚠️ **`segv/` 段错误 16 组对照** + 复现用探针脚本） | 同上 |
| `prototype/probes/` | **跨后端共用探针**（多流 16 项基线） | 命名规范见复核清单 §5 |

> **当前阶段主线汇报材料**：`prototype/docs/REFERENCE_TWO_INSTANCES_CONFIG_20260920.md`；
> **历史全量进度**：`P800/docs/PROGRESS_REPORT_20260914.md`；
> **上游镜像诉求**：`prototype/docs/IMAGE_REQUIREMENT_SPEC_20260920.md`。

---
### 6.6 看板归档与证据索引（**2026-10-08 看板精简时新增**）

> 背景：看板此前承载了大量**过程细节与我方缺陷复盘**，越积越杂。2026-10-08 做了一次精简 ——
> **看板只留「结论一行 + 报告与行号」**，过程、逐条判据、失败现场一律下沉到下列文件（**原文未改动**）。
> ⚠️ 判读纪律：归档里的**结论可能已被后续轮次更正**，引用前先看现行看板对应条目。

| 文件 | 内容 | 来源（原位置） |
|---|---|---|
| `prototype/docs/BOARD_ACTION_LOG_202608_202610.md`（76 行） | **主看板「近期动作」全量日志（60 项）** | 主看板 §5 |
| `prototype/docs/BOARD_ARCHIVE_README_HISTORY_202608_202609.md`（441 行） | **主看板历史原文**（2026-08 ~ 09 逐批记录；**旧目录结构**） | 主看板末段 |
| `prototype/docs/BOARD_ARCHIVE_STATUS_HEAD_CHAIN.md`（355 行） | **STATUS 的「上一版头部」历史链（8 版）** | STATUS 头部 |
| `prototype/docs/BOARD_ARCHIVE_STATUS_SECTIONS.md`（458 行） | **STATUS 的「当前阶段 / 已有量化结果 / 阻塞事项」原文**（含已解除阻塞与旧数字） | STATUS §当前阶段·§量化·§阻塞 |
| `prototype/docs/BOARD_ARCHIVE_PROTOTYPE_VALIDATION.md`（79 行） | **prototype 看板验证明细**（逐腿细节 · 对称复跑 · 历史缺口） | `prototype/README.md` §3 |
| `910C/docs/EVIDENCE_INDEX_910C.md`（41 行） | **910C 证据索引**（`probes/` 全量清单） | `910C/README.md` §4 |
| `P800/docs/EVIDENCE_INDEX_P800.md`（67 行） | **P800 证据索引**（探针脚本 + 逐份日志/JSON） | `P800/README.md` §5 |
| `MLU590/docs/EVIDENCE_INDEX_MLU590.md`（19 行） | **MLU590 证据索引** | `MLU590/README.md` §5 |

**同轮补入本目录的专题报告**（此前只在芯片看板 / 动作日志里被引用，未进总目录）：

| 文档 | 回答什么 |
|---|---|
| `prototype/docs/OPEN_ITEMS_AUDIT_20260929.md`（405 行） | **未收尾项回溯核查**：A/B/C/D 四类逐项清单 + 排序建议 + **明确不计入的项** + 一键复核命令。**当前未收尾口径以它为准**（A/B/C/D 已全部收尾；D3/C4 待定） |
| `prototype/docs/ENV_AND_HOSTCOPY_CLOSURE_C2_C3_20260930.md`（117 行） | **C2 宿主副本对齐 + C3 环境收口**：910C 宿主副本从 `856b24e`（落后 208 提交 / 109 脏文件）对齐到远端 tip；旧副本整目录留档（48 GB，未删文件、stash 可回滚） |
| `MLU590/docs/MLU590_FIX_WORKPACK_20261008.md`（188 行） | **MLU590 修复/补齐工作包**：差距清单 · 环境前置四条 · 一次窗口的执行序列 · **"跑的是当前版本"复核命令** · 已裁定项（职责审计扩到 §1.10） |
| `P800/docs/KUNLUN_P800_SHARED_LAYER_RERUN_20260930.md`（138 行） | **P800 r6 共享层复跑**：共享层改动（`release_stream` / 校验收紧 / 探针）的补齐覆盖；含 1 处判据口径不一致的修复 |

---

