# 运行时层接口约定 · 设备上下文章节
## —— flagos-runtime v0.1.0（2026-09-08 定稿）

> **读者**：运行时层各子方向（显存/分布式/监控/精度/算子/调度）及上层算子层、编译层的对接人。
> **效力**：本文档是设备上下文方向的**接口承诺**；下游按本文档接入，接口行为以本文档为准。
> **配套**：组件说明 `runtime/README.md` ｜ 基座配置 `dev/stack.lock.910c.v2.yaml`

---

## 1. 统一 API 承诺（下游直接使用）

### 1.1 后端选择

```python
runtime.use("ascend")     # 选择后端；换芯片只改这一行
runtime.available()       # ["ascend"] —— 当前可用后端
runtime.discover()        # 扫描已安装的 backend 插件
```

- 语义：`use()` 之后，全部设备操作路由到所选后端；**同一份业务代码不因换芯片而修改**
- 约束：`use()` 必须在任何设备操作之前调用；重复调用视为切换（需重新 set_device）

### 1.2 设备

| 接口 | 签名 | 语义 |
|---|---|---|
| `device_count()` | `→ int` | 可见设备数 |
| `set_device(ordinal)` | `ordinal: int` | 绑定当前设备；后续操作默认在此设备 |
| `memory_stats()` | `→ dict` | 显存统计（total/used/峰值等字段，显存方向经此采集，口径以此为准） |
| `probe_device(ordinal)` | `→ bool` | 设备探活（轻量，不干扰业务） |

### 1.3 流与事件

| 接口 | 语义 | 关键约束 |
|---|---|---|
| `create_stream()` | 创建统一 Stream 对象 | 跨流传缓冲必须 `record_stream`（见 §3 纪律 1）；**2026-09-29 起该纪律有实现路径约束**：先判能力位 `supports("record_stream")`，见 §1.6 |
| `create_event()` | 创建统一 Event 对象 | record 后 query 才有意义 |
| `stream.wait_event(ev)` | 建立跨流依赖（A record → B wait → B 可见 A 的结果） | 事件必须先 record |
| `stream.synchronize(timeout_ms)` | **有界**流同步；超时抛 `TimeoutError` | 长驻服务必须用有界同步，防整体 hang |
| `event.wait_host(timeout_ms)` | **有界**主机侧等待 | 同上 |
| `stream_ctx` | 流上下文（with 语法） | 上下文内的操作入该流 |

### 1.4 错误翻译

```python
fe = runtime.translate_error(exc, location="...")
# fe.category      → L1_RESOURCE / L2_PARAM / L3_EXECUTION / L4_FATAL
# fe.disposition   → retry / raise / replay / device_recovery
# fe.mapped        → 是否命中已知错误码映射
# fe.graded_by     → 分级来源（code_map=映射表 / message_hint=消息规则）
```

**处置约定**（下游必须按 disposition 处理，禁止按错误消息字符串自行判断）：

| 分级 | 含义 | 处置 | 责任方 |
|---|---|---|---|
| L1_RESOURCE | 资源类（显存不足等） | 重试或交显存方向扩容 | 显存/设备 |
| L2_PARAM | 参数类 | **上抛调用方**（重试无意义） | 调用方 |
| L3_EXECUTION | 执行类（如流同步超时 507046） | 重放（有机会成功）+ 调度重排 | 设备+调度 |
| L4_FATAL | 芯片级致命（如 AICORE 异常 507015） | **设备级恢复** `recover_device` + 检查点恢复 | 设备+监控+分布式 |

> **码表归属（2026-09-29 补，修正实测缺陷）**：**错误码表按厂商归属**。只有后端**声明了 `error_map`**
> （拥有本厂商码表）时，才允许依据码表把错误升级到更高分级；**未声明者不得使用任何他厂码表**
> 参与分级判定。**无依据时的正确行为是兜底 `L3_EXECUTION`**。
> 实测缺陷：未声明 `error_map` 的后端曾因异常消息里携带**昇腾**码 `507015` 被判
> `L4_FATAL` / `device_recovery` —— 而本节明令「下游必须按 `disposition` 处理」
> ⇒ 会**误触发设备级重建**（最昂贵的动作）。修复方式：译码器增加 `vendor_codes` 开关，
> 无本厂商码表的后端**从源头**不查该表，使 `graded_by` / `mapped` / `error_code` / `category`
> **四者结构上不可能不一致**（取代原先逐字段事后降级的脆弱写法）。

### 1.5 状态恢复

| 接口 | 语义 |
|---|---|
| `device_state(ordinal)` | 设备四态查询（AVAILABLE / DEGRADED / ISOLATED / **DESTROYED**） —— 四态成员名以 `conformance/device_state.py::DeviceState` 为实现基准（见变更记录 2026-09-28） |
| `recover_device(ordinal, mode)` -> **dict** | 三级重建：`probe`（保底探活）/ `real`（CANN 官方 aclrtResetDevice 序列）/ `hybrid`（先 probe 后 real） |

- **`recover_device` 返回的 `state` 取值域（2026-09-29 补）**：**必须是四态规范 token** ——
  `available` / `degraded` / `isolated` / `destroyed`（即 `DeviceState` 的 `.value`；
  实现基准 `runtime/backends/base.py::DEVICE_STATE_TOKENS`）。
  ⚠️ 此前三家都写 `str(state)`，得到的是 `'DeviceState.AVAILABLE'` 这类 **`str(enum)` 形式**，
  **不在**取值域内 ⇒ 下游按契约比较 `state == "available"` 会**判假**。
  注意：`device_state()` 返回**枚举对象**，而 `recover_device()["state"]` 返回**token 字符串**；
  两端消费者请按契约取 token 字符串这一形态。
- **调用约定**：监控方向做检测与恢复编排（何时调、调哪级），恢复执行由本组件完成
- **约束**：L4 级错误流级重试无效，必须走 `recover_device`；real 模式当前默认不启用
  （本地多进程联调已过，生产默认前需多卡压测调优——见 9 月计划 W4 遗留）

---

### 1.6 内存句柄与生命周期（**工作包 B，2026-09-29 新增 · 只增不改**）

| 接口 | 语义 | 关键约束 |
|---|---|---|
| `allocate(size_bytes, ordinal=0)` | 申请设备内存，返回**统一句柄** | 未声明 `memory_alloc` ⇒ **如实报错**；`size_bytes` 非正整数 ⇒ `ValueError`；**只做句柄语义，不含池化 / 碎片 / 峰值 / 扩容策略**（那些属显存方向） |
| `free(handle)` | 释放句柄 | **成对释放**；**二次释放 / 非本层句柄 ⇒ `ValueError`**（不得静默）；**不接受厂商裸指针/原生句柄** |
| `memory_handle_count()` | 当前在世句柄数 | 泄漏判据的取数入口 |
| `memory_stats()` | 规范**三键不变** +（可选）`allocated_mb` | 三键语义与口径不变；`allocated_mb`=**分配器视角**，仅当后端**声明 `memory_alloc_stat`** 时承诺给出，**取不到须省略该键**（不得填 0 冒充）；不得出现未登记键 |
| `Stream.record_stream(t)` | 跨流内存保护（§3 纪律 1 的实现路径） | 先判能力位 `supports("record_stream")`；后端未声明或张量不支持 ⇒ **保守同步路径**（同步当前设备后放行）+ 告警 + 计入 `degradations`，**不抛错**（跨流内存被提前回收是"偶发数据错乱"，比降级慢更危险） |
| `info()["native_accesses"]` | `.native` 逃生舱取用审计 | **公开属性 `.native` 计数**；**层内部路径不计数**（内部走私有口） |
| `info()["degradations"]` | 退化路径审计 | 与上者**语义相反、分开计数**：`.native` = 破坏可移植性；退化 = 用性能换正确性 |

**句柄公共字段**：`{handle_id, kind, backend, ordinal, size_bytes}`（`MEMORY_HANDLE_KEYS`）。
**承诺：句柄中不含厂商指针** —— 需要绕过统一层请走 `.native` 显式逃生舱，并计入审计。

### 1.7 设备上下文生命周期（**工作包 C，2026-09-29 新增 · 只增不改**）

> ⚠️ 澄清一处长期混淆：`stream_ctx`（§1.3）是**切流**的上下文管理器，
> 与本节"设备上下文（context）的创建/销毁"**是两件事**。

| 接口 | 语义 | 关键约束 |
|---|---|---|
| `context_create(ordinal=0)` | 新建设备上下文并**置为当前**，返回统一句柄 | 未声明 `context_lifecycle` ⇒ **如实报错** |
| `context_set(handle)` | 切换为当前上下文 | **只接受本层句柄** |
| `context_destroy(handle)` | 销毁上下文 | ⚠️ **只接受本层创建的句柄**；对非本层句柄（**尤其是进程默认上下文**）**一律拒绝并报错** —— 误毁默认上下文会让整个进程的设备不可用 |
| `context_count()` | 本层在世上下文数 | **不含**进程默认上下文 |
| **`context_query()`**（2026-09-29 第五轮补） | 查询**此刻实际生效**的设备上下文（**只读**） | 与 `context_lifecycle` **分开声明** —— 有的栈能管生命周期、有的栈**只允许观测**（平台单上下文）。返回**固定 6 键** `{queryable, present, ordinal, flags, managed_by, reason}`；**未声明者也返回同一 6 键**（`queryable=False` + **具体原因**）⇒ 上层换芯片无需分支。`managed_by ∈ {"unified", "external"}` 把「**归谁管**」写进字段 |
| **绑定语义** | 上下文销毁后，其上创建的流/事件**即为无效，不可继续使用** | 使用点（`Stream.context()` / `Stream.synchronize()`）**必须如实报错**。⚠️ 厂商栈在此处**不会立刻报错**（实测 910C：直到进程退出清理阶段才暴露 `stream not in current ctx` / 107003）⇒ **拦截由本层负责** |
| `recover_device()` 增键 | `context_supported` / `context_count` / `context_recreated` | **只增不改**：契约五键 `{ordinal, mode, recovered, state, detail}` **保持不变**；不支持上下文的后端如实置 `False` / `None` |

**句柄公共字段**：`{handle_id, kind, backend, ordinal}`（`CONTEXT_HANDLE_KEYS`）——
与内存句柄**同一套 `handle_id` 命名**、用 `kind` 区分种类；
**两类句柄不得互相误用**（误用必须报错，不许静默）。

**为什么 `context_query` 必须与 `context_lifecycle` 分开**（P800 实测）：
P800 的驱动层**有完整的 `cuCtx*` 系列**（21 个，就在 XPytorch 用的 `libcuda.so.1` 里），
但 **① 平台只允许一个上下文**（第二次 `cuCtxCreate_v2` 返回 `rc=2`）、
**② 该上下文由 XPytorch/XRE 自建**、**③ 本层抢先去建会破坏框架**
（实测 torch 报 `CUDA error: invalid device ordinal`；销毁本层的上下文后 torch 立即恢复）
⇒ P800 上「创建/切换/销毁/计数」四件事**全都不成立**，只能做**只读观测**。
把 P800 也标成支持 `context_lifecycle` 就是把不成立的事实说成成立。

**`managed_by` 的定位（2026-09-29 第五轮末补）**：它是「**尽力而为的归属提示**」，**不是判据**。
取值由「厂商上下文对象 vs 本层登记表」比对得出；实测（910C）在**单设备紧邻调用**下可靠
（建后 `unified`、销毁后 `present=False`），但在**多设备反复切换**的时序下
`create_context` 与 `get_context` 的返回值可能不一致，此时可能误报 `external`。
⇒ 依赖该字段做**资源回收/销毁**决策属误用（销毁口另有「只接受本层句柄」的硬校验兜底）。
多设备大规模混用前需补测。

**证据与判据**：报告 `WORKPACKAGE_BC_INTERFACE_20260929.md` §11、
`../../P800/docs/KUNLUN_CONTEXT_SEMANTICS_20260929.md`（四组判别实验）；真机证据
`../../910C/probes/probe_bc_contract_ascend_20260929.json`、
`../../P800/probes/probe_bc_contract_kunlun_20260929.json`、
`../../P800/probes/probe_bc_contract_kunlun_20260929_r2.json`（含 C4 组）。

---

### 1.8 契约不变式 I1–I4（**2026-09-29 定名并落地判据** · 只增不改）

> **为什么补这一节**：`VERIFICATION_MANIFEST` §1 第 9 条与 §3 G8 早已登记
> 「契约不变式（I1–I4）**未实现**，目前靠人工检查」—— 但**四条一直只有名字、没有定义**
> （全仓 `grep` I1–I4 只命中两处引用，且都只解释 I2）。本节把定义补齐，并注明每条**可追溯的来源条款与首例**。
>
> 四条不变式约束的**不是新 API，而是既有 API 的可观测性** —— 即「本层说的话是否可信」。
> 它们对**任何后端、任何芯片**都必须成立。

| # | 不变式 | 一句话 | 来源条款 / 首例 |
|---|---|---|---|
| **I1** | **诚实声明** | 声明的能力必须真的可调用；**未声明的必须显式拒绝**（不得静默假装可用） | 契约「声明即承诺」；台账第 ② 条（补实现不删声明） |
| **I2** | **禁止伪造** | 可观测字段之间**不得自相矛盾**；不得把"无依据的结论"呈现为**有依据** | 台账 §2.3（`mapped=True` 同时 `graded_by=message_hint_*` ⇒ 记录自相矛盾）；第 ③ 条 |
| **I3** | **失效受管** | 已失效对象（句柄 / 上下文及其流）**不得静默可用**，必须在**使用点如实报错** | 契约 §1.6 / §1.7；工作包 C 绑定语义（910C 实测厂商侧**当场静默**） |
| **I4** | **降级可观测** | 任何降级必须**计数可查**；`.native` 取用与退化路径**分开计数**（两者语义相反） | 修订建议 §3；工作包 B-3 / B-4 |

**判定细则（判据照此实现，不得自行放宽）**

- **I1**
  ① `supports(k)` 对**能力全集内**的键返回**严格 `bool`**；
  ② 能力全集**之外**的键一律 `False`（**不得 `True`**）；
  ③ `info()["capabilities"]` == 排序后的**声明集** `_capabilities`，⊆ 能力全集、**无重复**；
  **规范键**：`supports(k) is True ⟺ k ∈ capabilities`；
  **别名键**（`_CAPABILITY_ALIASES`）：取值随规范键，且**别名本身不得出现在 `capabilities`**
  （别名是**兼容入口**、不是在册能力 —— 否则同一份下游代码会在不同芯片上读到不同的在册集合；
  2026-09-29 依此判据发现并修复台账第 18 条）；
  ④ 声明为 `True` 的能力，其**公共入口必须存在**；声明为 `False` 的能力，其入口若存在则必须
  **显式拒绝**（抛错，**或**返回带 `queryable=False` + 具体原因的显式结构）
  —— **不得静默返回成功，也不得返回伪造值**。
- **I2**
  ① 对**不含任何已知关键词**的异常：`mapped` 必须为 `False`、`error_code` 必须为空；
  ② `mapped is True` ⇒ `graded_by` 必须属于「码表 / 规则命中」集合 —— **二者不得相反**；
  ③ `category` / `disposition` / `state` / `present` 只能取**已登记取值域**内的值
  （`present` 必须是 `bool` 或 `None`，**不得用 `0/1` 冒充**）；
  ④ **取不到就必须缺席**：如未声明 `memory_alloc_stat` 时，`memory_stats()` **不得**出现 `allocated_mb`
  （不许填 0 冒充）。
- **I3**：**二次释放** · **二次销毁** · **跨种类误用**（内存句柄当上下文销毁 / 反之）·
  **销毁上下文后使用其流** —— 四类都必须**如实报错**；能力未声明时如实标「不适用」。
- **I4**
  ① `degradations()` / `native_accesses()` 结构为 `{total, by_kind}`，取值**单调不减**；
  ② 经 `.native` 取用一次 ⇒ `native_accesses().total` **+1** 而 `degradations().total` **不变**；
  ③ `info()` 必须**同时暴露**二者（降级可观测的对外形态）。

**证据与判据**：真机判据 `runtime/conformance/contract_invariants.py`
（`runner.py --backend $B --cases contract_invariants` ⇒ `CONTRACT_INVARIANTS_PASS 4/4`）；
离线桩判据 `scripts/backend_offline_check.py` 第 `[10]` 段（同一套核心函数，可在无设备时拦住回归）。

**边界**：四条都是**可观测性**约束，**不评判性能**；判据遇"能力未声明"时如实标**不适用**
（计为通过但**注明不适用**），**不得**因不适用而把未声明能力当作已验。

**状态（2026-09-29）**：**已实现**。**离线桩已通过**（`backend_offline_check.py` 第 `[10]` 段，与真机**同一套核心函数**：`ascend 75/0/1` · `kunlun 76/0/1` · `cambricon 64/0/0`）；**5 处注入的非空转验证全部被抓**。⏳ 真机复跑（`--cases contract_invariants`）安排在与第 5 轮全套回归（r5）同窗口。报告：`CONTRACT_INVARIANTS_I1_I4_20260929.md`。

---

## 2. Backend 插件接入规范（新芯片方向照此实现）

**新增一家芯片 = 实现一个 backend + 跑通 conformance。** 步骤：

1. 新建 `runtime/backends/<vendor>/`，实现 `RuntimeBackend` 抽象（`backends/base.py`）：
   - **五域抽象**：设备（count/set_device/memory_stats/probe）、内存（分配/释放/统计）、
     Stream-Event（创建/等待/有界同步/上下文）、错误码翻译、状态恢复
   - 三个多流支撑方法：`stream_context / synchronize_stream（有界）/ wait_event_host（有界）`
   - 可选能力用 `supports()` 声明（conformance 自动生成 stub-skip 报告）
2. 提供 `build()` 工厂函数，在 `registry` 登记名字
3. 跑通 conformance：`python3 runtime/conformance/runner.py --backend <vendor>`
   ——13 例 + 6 例全绿（或 stub-skip 报告说明缺口）即接入完成
4. 接入成本目标：**≤5 人天**（11 月昆仑芯/寒武纪实做验证）

> **服务启动不在本文档规定范围内**（指针，2026-09-20 补）：推理服务的启动脚本与参数
> 统一遵循《组内服务启动标准》`docs/SERVICE_STARTUP_STANDARD_20260920.md` ——
> 唯一入口 `scripts/serve_standard.sh`，跨芯片只改 `DC_BACKEND`，**各方向不要各自维护启动脚本**。
>
> 本文档对该流程只承诺与**接口语义**相关的部分：服务侧异常须经 `translate_error` 统一分级、
> 按 `disposition` 处置（见 §1.4）；启动参数口径的变化**不构成接口变更**，不走本文档的变更流程。
> 这样切分的理由：接口语义承诺需"知会全部下游 + conformance 回归"，而启动参数会随实例与镜像
> 更频繁演进，两者混在一起会让接口评审被启动参数变更污染。

---

## 3. 两条硬纪律（多流正确性，违反即数据错乱）

1. **`record_stream` 跨流保护**：张量缓冲在流 A 分配、交流 B 使用时，必须
   `buffer.record_stream(stream_B)`，告知缓存分配器该缓冲仍在被使用，
   否则可能被提前回收重用 → 数据竞争。（统一 Stream 已封装，直接调用）
2. **错误隔离分层**：API 级错误（如 107015）只影响该次调用，其他流不受影响，可流级重试；
   **芯片级错误（AICORE_TIMEOUT/EXCEPTION）影响该设备全部流**，流级重试无效，
   必须走设备级 `recover_device`。

---

## 4. 版本与兼容承诺

| 版本 | 承诺 |
|---|---|
| v0.1（当前） | 允许破坏性变更（提前一周知会下游）；下游锁定 `use("ascend")` 用法不变 |
| v0.1.x | 吸收 9 月下游反馈，不改已有接口签名 |
| v0.2（10 月） | 20 模型反馈迭代；新增能力不影响已有调用 |
| v1.0（2027.06） | **稳定接口承诺**：变更需评审 |

**变更流程**：接口变更 → 本文档更新（变更记录节）→ 知会全部下游 → conformance 回归全绿。

### 变更记录

| 日期 | 版本 | 变更 | 知会 |
|---|---|---|---|
| 2026-09-08 | v0.1.0 | 初版定稿（API 面 + 插件规范 + 两条纪律） | 运行时层全组 |
| 2026-09-20 | v0.1.0 | 在 §2 末尾新增**服务启动指针**：启动流程统一遵循《组内服务启动标准》，并明确"启动参数口径变化不构成接口变更"（接口版本不变，仍为 v0.1 原型期） | 运行时层全组 |
| 2026-09-28 | v0.1.0 | **三实例职责响应审计**（逐 sub-part 实测，见 `PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md`）发现并修正三处**文档↔实现**不一致：① §1.5 四态命名 `UNKNOWN` 与实现（三家共用 `DeviceState`）不符 → **以实现为准更正为 `DESTROYED`**（**行为无变化**，仅措辞对齐）；② `recover_device` 返回契约**五键**（`{ordinal, mode, recovered, state, detail}`）此前仅 `ascend` 齐全 → 已补齐另两家；③ `sync_timeout` 明确为 `bounded_sync` 的**弃用别名**（三家键集合对齐，取值随规范键）。接口签名**未变** | 运行时层全组 |

---

| 2026-09-29（第三轮） | v0.1.0 | **工作包 B/C 接口落地（只增不改）**：新增 **§1.6 内存句柄与生命周期**（`allocate/free/memory_handle_count`；`memory_stats` 只增 `allocated_mb`；`record_stream` 能力位与**保守同步路径**；`.native` / 退化**双审计**）与 **§1.7 设备上下文生命周期**（`context_create/set/destroy/count`；**绑定语义**由本层在使用点拦截；`recover_device` 只增 context 三键）。**未改任何既有接口签名**；四家 `_CAPABILITY_KEYS` 各新增 4 项键（未声明即如实为 `False`）。判据：离线自检新增 `[9]` 段、6 条负向判据收紧为「必须是契约级 `ValueError`」，并做 **5 处注入的非空转验证**；真机验证 **910C 6/6 · P800 3/3**（C 项在 P800 **如实不具备**）；**MLU590 本轮未探测**（4 个新键未声明）。报告：`WORKPACKAGE_BC_INTERFACE_20260929.md` | 运行时层全组 |
| 2026-09-29 | v0.1.0 | **「分歧的业务代价」实验**（工作包 A，见 `EXP_DIVERGENCE_COST_20260929.md`）暴露两处**已存在而未判据守**的缺陷并修正：① **码表归属** —— 未声明 `error_map` 的后端其 `category` 仍取自外来（昇腾 ACL）码表 ⇒ 携带昇腾码的消息被判 `L4_FATAL`/`device_recovery`；已改为**从源头不使用外来码表**（`translate_error(..., vendor_codes=False)`），使四个字段结构上不可能不一致；② **`state` 取值域** —— `recover_device()["state"]` 原为 `str(enum)`（`'DeviceState.AVAILABLE'`），已归一为四态规范 token（`base.state_token()`）。同时给离线自检补 2 条判据（**决策字段**不得受外来码表影响；`state` 取值域），并做**非空转验证**。接口签名**未变**；`translate_error` 新增的 `vendor_codes` 为**关键字参数、默认 `True`**，对既有调用完全兼容 | 运行时层全组 |
| 2026-09-29（第五轮） | v0.1.0 | **新增 `context_query`（上下文**只读观测**，见 §1.7）** —— **只增不改**，与 `context_lifecycle` **分开声明**。起因：P800 的 C 项原记为「兼容层未暴露上下文原语」，实测**更正**为该栈**有**完整 `cuCtx*`（就在 XPytorch 用的 `libcuda.so.1` 里）但**平台只允许一个上下文**、且由框架自建 ⇒ 生命周期四件事均不成立，只能只读观测。落地：`context_query()` 固定 6 键 + `managed_by ∈ {unified, external}` 把差异写进字段；四家 `_CAPABILITY_KEYS` 各加 1 项键（**ascend 与 kunlun 声明**（第五轮同日完成 910C 对齐）；**cambricon 如实未声明**且给出具体原因，MLU590 的探测仍挂账）。判据：离线自检 **+6 条**（含 2 条「有上下文」分支，用**可控桩**覆盖 —— 非空转验证证明该分支原先**抓不到缺陷**）；真机 **C4 组通过**（`managed_by="external"` / `readonly_safe=true`）。报告：`../../P800/docs/KUNLUN_CONTEXT_SEMANTICS_20260929.md` | 运行时层全组 |
| 2026-09-29（第二轮） | v0.1.0 | **跨实例复验再修一处判据覆盖缺口**：共享消息规则表的 L2 规则原按"个别厂商文案"枚举（`invalid (device\|ordinal\|data\|op\|param)`），寒武纪栈对"设备序号越界"的原文 `CNRT error: invalid argument.` **无规则命中 ⇒ 兜底 `L3_EXECUTION`（`replay`）**，与契约 §1.4 的「参数类应 `raise`」相悖 ⇒ 已改为**按等价类覆盖**（补 `invalid argument`／`invalid value`／`illegal …`，**不做** `invalid \w+` 宽匹配）。同时离线自检 +2 条「参数类文案等价类」判据（用两家真机原文）并做**非空转验证**。**接口签名未变**，`translate_error` 的签名与默认值均未动（仅内部消息规则表扩容） | 运行时层全组 |
| 2026-09-29（第六轮 · B1） | v0.1.0 | **新增 §1.8 契约不变式 I1–I4**（定名 + 定义 + 判定细则）—— 此前后者只有名字、没有定义（G8 登记为「未实现，靠人工检查」，全仓 grep 仅两处引用且都只解释 I2）。**只增不改**（不新增 API，只约束既有 API 的可观测性）。判据落地两处：真机 `runtime/conformance/contract_invariants.py`（`runner.py --cases contract_invariants`）+ 离线桩 `backend_offline_check.py` 第 `[10]` 段（**同一套核心函数**，无设备即可拦住回归）。G8 关闭 | 运行时层全组 |
| 2026-09-29（第六轮续 · B1） | v0.1.0 | **契约不变式判据落地**（G8 关闭）：判据落 `conformance/contract_invariants.py`（真机 4 例）+ `backend_offline_check.py` 第 `[10]` 段（离线，**同一套核心函数**按文件路径加载，避免两处漂移）。过程中：① 发现并修复**台账第 18 条**（`ascend` 把弃用别名 `sync_timeout` 当在册能力列出，另两家不列 ⇒ 同一份下游代码在不同芯片上读到不同的在册集合）；② 修正判据自身两处缺陷（I1③ 需**别名感知**、I1⑤ 需**只认契约级 `NotImplementedError`**，否则会把「抛任何异常」误当显式拒绝 ⇒ **假通过**）。**未改任何接口签名**，行为变化仅限 `info()["capabilities"]` 的归一 | 运行时层全组 |

---

## 5. 验收与支持

- **组件质量基线**：昇腾真机冒烟 37/37；conformance 13/13 + 6/6；跨天 28h 长驻零增长
- **下游接入自检**：接入后先跑 `python3 runtime/smoke_runtime.py`，全过即接入成功
- **问题反馈**：device-context（Kistich）；每周五前反馈的问题当周定位、下周版本修复
- **本周状态**：组件 v0.1 待打包下发（见 9 月计划 W2）；本章节即战略文档要求的
  "设备上下文接口约定章节"定稿

---

## 补充：recover_device 返回契约（2026-09-09 统一）

- **返回类型统一为 `dict`**：`{ordinal, mode, recovered, state, detail}`
- **`recovered` 语义 = 设备当前可用**（不是"是否执行了重建"）
  - 底层 `recovery.recover_device` 仅在设备处于 ISOLATED 时才执行重建，
    否则返回 False；此前 ascend 后端直接透传该 bool，导致"设备正常、无需重建"
    被上报为"恢复失败"。现已统一：以设备状态 + 探活结果判定。
- `detail` 区分三种情况：重建成功 / 无需重建（探活可用）/ 恢复失败（探活不可用）
- `state` 为设备四态之一，便于上层与监控方向判定

---

## 补充：错误分级调用纪律（2026-09-09 实测）

**调用 `translate_error` 时必须传入完整的原始异常 / 服务错误消息，不得截断。**

实测：vLLM 服务对超长输入返回 HTTP 400，错误体含
`"type":"BadRequestError","param":"input_tokens"` 与完整 message。
- 传完整 message → 正确分级 **L2_PARAM / raise**（参数类上抛，重试无意义）
- 仅传 "HTTP 400"（或错误体被截断导致 JSON 解析失败）→ 退化为 **L3_EXECUTION / replay**
  —— 语义丢失导致保守误判，会让上层对参数错误做无意义的重放。

建议：
- 服务端透传 vLLM 异常类型与 message，不要只留状态码；
- 客户端调用 `translate_error` 时把原始异常对象（或完整消息）传入，不做截断；
- 若只能拿到状态码，需显式使用状态码→分级映射，而不是交给消息兜底。
