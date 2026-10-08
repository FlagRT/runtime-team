# MLU590（寒武纪 `cambricon`）· 修复/补齐工作包（2026-10-08 制订）

> 作者：Kistich（hliu553）｜定位：**第三家实例的补齐计划**，供网络恢复后**一次窗口**执行。
> 前置状态：① 第 1 家 910C ✅、第 2 家 P800 ✅（见 §1）；② **MLU590 两台主机当前不可达**（见 §2）。
> ⚠️ 本文件是**计划**，尚未执行；所有数字都是"预期"或"历史实测"，执行后须以新证据回填。

---

## 0 结论先行

| 家 | 职责验收 | 当前版本覆盖 | 结论 |
|---|---|---|---|
| **910C**（`ascend`） | ✅ `DUTY_RESPONSE_PASS` **39 / 0 / 0** | ✅ **2026-10-08 第十轮全量复跑（20 项全绿）** | **已完成** |
| **P800**（`kunlun`） | ✅ `DUTY_RESPONSE_PASS` **36 / 0 / 3** | ✅ 2026-09-30 r8（18 项全绿，版本 `ea503cc`）；此后未改动任何原型代码 ⇒ 结论仍有效 | **已完成** |
| **MLU590**（`cambricon`） | ✅ 曾 `36 / 0 / 3`（**2026-09-28**，**旧口径 39 项**） | ❌ **停在 2026-09-29**；此后契约新增 5 章、共享层多次改动、新探针 6 类**一份都没跑过**，且**职责口径已扩到 78 项**（见 §0.1） | **待做 ⇒ 本工作包** |

### 0.1 ⭐ 口径更新（2026-10-08 · 本工作包已同步最新口径）

在制订本工作包后，同日又完成了**台账 E1（职责审计扩口径）**，直接影响本实例的上机清单：

| 变化 | 对 MLU590 的影响 |
|---|---|
| **职责审计 39 → 78 项**（新增 H 内存句柄 8 · I 设备上下文 10 · J 契约不变式 4 · K 流优先级 8 · **L 流所有权与释放 6** · M 统一 API 面 3） | §3 第 1 组的第 7 项按 **78 项**跑；**绝不要**拿旧的 36/0/3 当基线 |
| 新增**非空转验证**脚手架 `probes/selfcheck_duty_audit_ext.py` | §3 第 1 组新增一项（**必跑**）：逐条注入缺陷证明判据能 FAIL |
| 契约 §1.7 `context_set` / §1.9 `stream_priority_range` **已补为统一面出口**（台账第 30 条） | 本机应**直接通过** M1（若 FAIL ⇒ 说明同步的是旧原型） |
| ⭐ **侵入项子进程隔离**：审计已把 `I1/I4/I5/I10/J3` 放**子进程**执行 | 本机若声明 `context_lifecycle`，这五项会各起一个子进程（耗时略增）；**不要**改回同进程 —— 同进程内做过上下文 create/destroy 会让后续判据被污染（910C 实测：I8 误报 FAIL、第二次 `check_i3` 段错误 rc=139） |
| ⭐ **本实例是三家唯一会真正行使 K6 / L5 的** | 区间 `(0,-3)` **非单点** ⇒ **K6**（端点请求必须回读一致）与 **L5**（「由本层拥有」的流释放后再用必须 `RuntimeError`）**首次被真正验证**；K8 因非单点而如实 SKIP |

**2026-10-08 两家现行结论（供本机对照，不得当作本实例结论）**：
910C `DUTY_RESPONSE_PASS` **73 / 0 / 5** · P800 **67 / 0 / 11**（共 78 项）；
非空转 **910C 34 抓到 / 5 不适用 / 0 未抓到 · P800 31 / 8 / 0**。

> "910C 与 P800 已修复完成" 的**硬依据**：910C 今日实测；P800 的 09-30 轮；且此后的提交
> `898536a` **未触及任何原型代码**（`prototype/runtime|scripts|conformance` 命中 0 个文件）
> ⇒ 无新增破坏面 ⇒ 无需为它复跑。

---

## 1 MLU590 的差距清单（本工作包要收的东西）

以 `MLU590/probes/` 现有证据为准（`ls` 计数），**以下 6 类探针在该实例上是 0 份证据**：

| 缺失项 | 910C / P800 是否已有 | 为什么它对 MLU590 特别重要 |
|---|---|---|
| **B/C 契约探针** `probe_bc_contract.py` | ✅ / ✅ | 本实例的 **`memory_alloc` / `memory_alloc_stat` / `record_stream` / `context_lifecycle` / `context_query` 五个能力键全部未声明**（= 从没探测过，**不等于"确认不具备"**）⇒ 取数后才谈得上声明与否 |
| **契约不变式** `--cases contract_invariants` | ✅ 4/4 / ✅ 4/4 | 它守的正是本项目反复踩的家族（假绿 / 静默退化 / 失效对象可继续用），**本实例从未跑过** |
| **公开入口定向验证** `probes/recover_entry_verify.py` | ✅ / ✅ | 验证「某卡 L4 故障 → 设备级恢复」在公开面上真能触发（`set_device_state` / `handle_error`） |
| **优先级 API 探针** `probes/probe_stream_priority_api.py` | ✅ 8/8 / ✅ 6~7 | ⭐ **本实例是三家唯一声明「能设置」的**（`stream_priority_control` + `stream_priority_readback`），但**该声明的依据是 2026-09-22 的真机记录**，而本层新接线（三段校验 + 强制回读）**从未在本实例跑过** ⇒ 这正是台账 **D1** |
| **流所有权/释放探针** `probe_stream_release_and_control.py` | ✅ / ✅ | 验证「厂商拥有的流 release 必须 no-op」「本层拥有的流必须真销毁、销毁后使用必须报错」 |
| **根解析自检** `probes/selfcheck_root_resolution.py` | ✅ 38/0 / ✅ 38/0 | 纯路径运算、**离线即可跑**，专抓 `ModuleNotFoundError: No module named 'runtime'` 这一整类（同类错曾一天犯两次） |
| （附）**多卡多进程 `real` 压测** | ✅ 3 rank×30 轮 / ❌ | 可选；本实例若声明 `recovery_real` 则应做（见 §4 注意事项） |

**另有两类"口径"要一并处理**（§5）：职责审计的**范围**是否扩到契约新增章节。

---

## 2 环境前置（网络恢复后先跑这四条）

> 实测不可达时间戳：**2026-10-08 10:57:47 CST**，`Mlu-1`(10.1.1.21) / `Mlu-2`(10.1.1.22)
> 均 `ssh: connect to host … port 22: Operation timed out`（TCP 层不通 ⇒ 是网络未启用，不是服务没起）。

| # | 项 | 期望 / 说明 |
|---|---|---|
| 0 | ⭐ **先验模型文件完整性** | 共享路径 `/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/<hash>/model.safetensors`：<br>① `sha256sum` 与 HF 的 blob 名比对（内容寻址自证）；② 用 `model_repair_20261008/weight_scan.py` 扫一遍非有限值。<br>⚠️ **这不是多余的**：2026-10-08 在 910C 上实测过**同名同源**的模型文件被存储层静默损坏（2 张量 34396 个非有限值）⇒ 失败会伪装成"本层回归" |
| 1 | 连通性 + 容器 | `ssh Mlu-1 'hostname; docker ps -a --format "{{.Names}}\|{{.Status}}"'`；容器 `dc-mlu590-hliu553`（运行时）/ `dc-mlu590-vllm-hliu553`（服务化） |
| 2 | 路径映射 | 宿主 `/srv/hliu553` → 容器 **`/work`**；宿主 `/srv/data/hf_cache` → 容器 **`/hf_cache`**（⚠️ 与 910C「同路径」不同，**别按 910C 猜**） |
| 3 | 解释器 / 档位 / 集合通信 | 解释器 `/flagos/bin/python3`；镜像档 **`neuware4.4.3`**（宿主驱动 6.2.29 ⇒ **只能**这一档）；集合通信后端名 = **`cncl`**（已实测 2 进程 `all_reduce` 正确） |
| 4 | 起服务形态 | **必须用 vLLM 应用镜像**（运行时镜像**不含** vLLM）：`flagos-app/vllm0.20.2-cambricon-neuware4.4.3:2.2.0-0.2.2rc2.post2` |
| 5 | 两个已登记的坑 | ① `import triton` **必须先 `import torch_mlu`**（否则触发失败加载）；② `/opt/triton` **只在经 shell 启动时**进 `PYTHONPATH`（`docker exec … python3` 直调看不到） |

---

## 3 执行顺序（一次窗口的目标序列）

> 顺序原则同前两家：**先无卡/短项 → 再两条腿 → 最后服务化**。

**第 0 步：把当前原型同步进去**（HEAD `898536a`）

```bash
rsync -az --exclude='__pycache__' --exclude='*.pyc' --exclude='out*' \
  dev/device-context/prototype/ Mlu-1:/srv/hliu553/dc_mlu_regen_20261008/prototype/
# 开工前先复核"跑的是当前版本"：grep 关键修复点（owns_stream / release_stream / [8b-②] 段）
```

**第 1 组 · 离线/短项（无多卡需求）**

| 序 | 命令（`--backend cambricon`；`DC_BACKEND=cambricon`） | 历史值（09-29） | 本轮期望 |
|---|---|---|---|
| 1 | `scripts/backend_offline_check.py --backend cambricon` | 45/0/0 | 判据数**会变**（新增 `[8b]`/`[8b-②]` 段）⇒ **0 失败** 即通过 |
| 2 | `scripts/backend_offline_check.py --all` | 对称性 5/0 | **0 失败** |
| 3 | `runtime/smoke_runtime.py --backend cambricon` | 46/0 | **0 失败** |
| 4 | `conformance/runner.py --backend cambricon` | 13/13 | `CONFORMANCE_PASS` |
| 5 | `conformance/runner.py --backend cambricon --cases infer_cases` | 6/6 | `CONFORMANCE_PASS` |
| 6 | `conformance/runner.py --backend cambricon --cases contract_invariants` | ⛔ **从未跑** | `CONTRACT_INVARIANTS_PASS` |
| 7 | `scripts/duty_response_audit.py --backend cambricon` | 36/0/3（**旧口径 39 项，不得当基线**） | **78 项**口径下 `DUTY_RESPONSE_PASS`，**0 FAIL**（SKIP 均为如实不具备） |
| 7b | `probes/selfcheck_duty_audit_ext.py --backend cambricon` | ⛔ 从未跑 | `SELFCHECK_DUTY_EXT_PASS`（**0 未抓到**）；本机不适用项换实例才成立，见 §0.1 |
| 8 | `runtime/proto/proto_error_recovery_loop.py --backend cambricon` | 5/0/0 | `ERROR_RECOVERY_LOOP_PASS` |
| 9 | `probes/recover_entry_verify.py --backend cambricon --dev 0` | ⛔ 从未跑 | `ENTRY_VERIFY_PASS` |
| 10 | `probes/selfcheck_root_resolution.py` | ⛔ 从未跑 | `ROOT_RESOLUTION_PASS`（正确数 = 38 或随探针增减） |
| 11 | `probes/probe_stream_semantics_full.py` | 8/8 | `STREAM_SEMANTICS_PASS` |
| 12 | `DC_QUOTA_N=2000 probes/probe_stream_quota.py` | 3/3 | `STREAM_QUOTA_PASS` |
| 13 | `runtime/demos/demo_unified.py` | —— | 演示跑通 |

**第 2 组 · ⭐ 三项"取数决定声明"的探针（本工作包的核心）**

| 序 | 探针 | 产出决定什么 |
|---|---|---|
| 14 | `probes/probe_bc_contract.py --backend cambricon --dev 0` | **5 个能力键**（`memory_alloc` / `memory_alloc_stat` / `record_stream` / `context_lifecycle` / `context_query`）**声明 or 如实不声明**；未声明一律**不得**凭"应该支持"补上 |
| 15 | `probes/probe_stream_priority_api.py --backend cambricon --dev 0` | **台账 D1**：三家唯一「能设置」是否成立（`create_stream(priority=…)` + **C API 回读**一致） |
| 16 | `probes/probe_stream_release_and_control.py --backend cambricon --dev 0` | 流**所有权/释放**语义（厂商流 release 必须 no-op；本层流 release 必须真销毁 + 销毁后使用必须报错） |

**第 3 组 · 两条腿**（须给 `DC_ROOT`，且 `DC_MODEL` 指向 §2 步 0 验过的模型）

| 序 | 项 | 历史值 | 说明 |
|---|---|---|---|
| 17 | 训练腿 2 卡 | `TRAIN_LEG_PASS 6/6`（loss 15.4498→11.1479；2957.8 / 3015.3 tok/s） | `DC_DIST_BT` 用 **`cncl`**；⚠️ 集合通信后端名**必须实测**，不得从另两家类推 |
| 18 | 推理腿前向 | `INFER_LEG_PASS 13/13`（另 1 项 `vendor_code_map` **如实跳过**） | dim 1024、区分度 0.6391 |
| 19 | （可选）多进程 `real` 压测 | ⛔ 从未跑 | 仅当本实例声明 `recovery_real` 时做 |

**第 4 组 · 服务化**（单卡 + **vLLM 应用镜像**容器）

| 序 | 项 | 历史值 |
|---|---|---|
| 20 | `SERVE_FORM=embed` 的 `serve_standard.sh` | `SERVE_STANDARD_PASS`（就绪 150 s、维度 1024、范数 1.000001） |
| 21 | `proto_infer_serve.py --backend cambricon`（本层消费方） | 覆盖 `create_stream()` 的最后一个消费方 |

**收尾**：① 停自己的带卡容器；② 查残留进程；③ 核他人容器未动；④ 证据落 `MLU590/probes/`。

---

## 4 预期会改动的代码面（只增，均由探针结果决定）

| 面 | 触发条件 | 改法 |
|---|---|---|
| `cambricon/backend.py` 的 `_capabilities` | §3 第 14/15/16 项的取数结果 | **只增**：实测支持的才声明；不支持的在注释里写清「**已实测确认不具备**」还是「**未验证**」（两者不可混写） |
| `known_issues` | 若探针暴露新的环境/厂商约束 | 只增，附证据与可执行诉求 |
| 三家如实矩阵 / 契约 §1.9 表 | 若第 15 项结论与 09-22 记录不同 | 回填并标注日期与批次；**历史值保留并标「首测」** |
| 契约不变式（若第 6 项 FAIL） | —— | 按"先确认判据与被测对象是不是同一个世界"的顺序排查，**不得改判据变绿** |

---

## 5 ✅ 已裁定并落地（2026-10-08）：职责审计的范围已扩到 §1.10

**事实**：`scripts/duty_response_audit.py` 的口径写死为
「接口约定 §1.1–§1.5 + §2（三支撑）+ §3（两纪律）」= **39 项**；
对该脚本 grep `priority` / `owns_stream` / `release_stream` ⇒ **0 命中**。

而契约在 2026-09-29/30 又新增了 **5 章**：

| 章 | 内容 | 现有判据落在哪 |
|---|---|---|
| §1.6 | 内存句柄与生命周期（`allocate/free/memory_handle_count`、`record_stream` 能力位） | 离线自检相应段 + `probe_bc_contract.py` |
| §1.7 | `context_query`（上下文只读观测） | 离线自检 + `probe_bc_contract.py` |
| §1.8 | 契约不变式 I1–I4 | `conformance --cases contract_invariants` + 离线 `[10]` 段 |
| §1.9 | 流优先级（`create_stream(priority=)` + 三个能力键 + 四道约束） | 离线 `[8b]` 段 + `probe_stream_priority_api.py` |
| §1.10 | 流所有权与释放 | 离线 `[8b-②]` 段 + `probe_stream_release_and_control.py` |

⇒ **判据都在，但不在"职责响应审计"里**，职责文档也没写这条映射
⇒ 读者会以为「39/0/0 = 全部职责已响应」，实际它只覆盖到 §1.5。

**裁定结果（2026-10-08）**：**选 A**（扩口径）—— 已完成并在 910C / P800 复跑（见 §0.1）。
下表的三个选项**保留作决策记录**。

**三个选项（原样保留）**：

| 选项 | 做法 | 代价 | 收益 |
|---|---|---|---|
| **A（推荐）** | 把 §1.6–§1.10 拆成新 sub-part **并入职责审计工具**（如 39 → 39+N），三家各补跑一次职责审计 | 需 910C / P800 **各补跑一次**（单脚本，几分钟，**不需要两条腿**） | 职责验收口径与契约**齐平**；**MLU590 一次窗口就覆盖全口径**，不必二次开窗 |
| **B** | 不动工具，只在三份职责文档 + 汇总里**登记覆盖映射**（新章由哪些套件覆盖） | 零代码风险，纯文档 | 消除"名义上无人认领"的误导；但职责审计仍只到 §1.5 |
| **C** | 都不做 | 0 | MLU590 按现状套件跑（口径仍停在 §1.5） |

> ⚠️ 若选 A，**必须在 MLU590 上机前**完成工具改动，否则寒武纪要跑两轮。

---

## 6 边界

- 本文件是**计划**：§0 的 910C / P800 结论是**实测**，MLU590 的 §3 各值是**历史实测（2026-09-29 及更早）**，
  **不是本轮结果**；执行后须以新证据回填，并保留历史值（标「首测」）。
- §2 的"不可达"结论**带时间戳**（2026-10-08 10:57:47 CST）；共享机网络是**间歇性**的，
  网络恢复不等于一次就能跑完，也不等于主机上的镜像/容器状态没变。
- MLU590 的 §3 命令按**前两家已跑通的同一套工具与模式**书写，**尚未在本机实测**
  （与前两家不同的是路径映射 `/work`、解释器 `/flagos/bin/python3`、集合通信 `cncl`、服务化镜像）；
  首次执行时若与预期不符，**以实测为准并回填本文件**。
