# 统一原型 · 职责响应审计（三实例 × 39 项 sub-part）

> **日期**：2026-09-28 ｜ **负责人**：Kistich（hliu553）
> **判据来源**：《运行时层接口约定 · 设备上下文章节》`INTERFACE_CONTRACT_DC_20260908.md`
> （§1.1 后端选择 / §1.2 设备 / §1.3 流与事件 / §1.4 错误翻译 / §1.5 状态恢复 /
> §2 三支撑方法 / §3 两条硬纪律 / 补充·`recover_device` 返回契约）
> **证据**：`910C/probes/duty_audit_ascend_20260928.json`、`P800/probes/duty_audit_kunlun_20260928.json`
> **工具**：`scripts/duty_response_audit.py`（新增，可对任意后端单跑，供后续新芯片复用）

---

## 一、为什么要做这次审计

已有的三套件各覆盖一部分，但**都没有回答一个问题**：

| 套件 | 它回答的问题 | 它的盲区 |
|---|---|---|
| `backend_offline_check.py` | 契约**形态**对不对（离线、桩） | 不回答「接口约定承诺的**每一个** API 是否都能顺利响应」 |
| `conformance`（13+6） | **行为契约**对不对（重点用例） | 用例是抽样，未覆盖的 API 无人过问 |
| `smoke_runtime.py` | 接入**可用性** | 同上，且偏"能跑起来" |

⇒ 需要一个**按接口约定逐 sub-part 点名**的审计：把 §1.1–§1.5 + §2 + §3 拆成
**39 个可执行 sub-part**，逐个调用并判定"**响应且合契约**"。

> ⭐ **2026-10-08 扩口径（台账 E1，用户裁定选 A）**：契约此后新增 §1.6–§1.10
> （内存句柄 / 设备上下文 / 契约不变式 / 流优先级 / 流所有权与释放），而那 39 项**未覆盖新章**
> ⇒ 已把 §1.6–§1.10 与「**统一 API 面**」一并拆成 sub-part 并入工具，**总项数 39 → 78**：
> 新增 **H 内存句柄与生命周期 8 · I 设备上下文 10 · J 契约不变式 4 · K 流优先级 8 ·
> L 流所有权与释放 6 · M 统一 API 面 3**（J 域**委托同一份实现** `conformance/contract_invariants.py`，
> 不另写一套，避免两处漂移）。

**判定口径**（三类，只有 FAIL 算缺口）：

| 标记 | 含义 |
|---|---|
| `OK` | 调用成功 **且** 返回值/副作用符合接口约定 |
| `FAIL` | 不响应（异常/未实现）**或** 响应但不符合契约（字段缺失、静默降级） |
| `SKIP` | 该能力被后端**如实声明为不具备**，且调用被主动拦截 —— **不是缺口** |

---

## 二、审计项构成（39 → **78** 项；2026-10-08 扩口径）

| 域 | 项数 | 覆盖内容 |
|---|---|---|
| A 后端选择（§1.1） | 4 | `discover` / `available` / `use`+`current`+单例 / **负向**：未注册名如实抛 `BackendNotFound` |
| B 设备域（§1.2） | 5 | `device_count` / `set_device` / `memory_stats` 结构 / **值自洽**（total≈used+free） / `probe_device` |
| C 流与事件（§1.3） | 14 | `create_stream/event`、`current_stream`、无界+**有界**同步、`stream_context`、**跨流依赖真实语义**、`wait_stream`、`event.record/query/wait/synchronize`、**未 record 的 `wait_host` 不永久阻塞**、`elapsed_time`、**硬纪律 1 `record_stream`** |
| D 错误翻译（§1.4） | 6 | 统一类型 / category 四类 / disposition 四动作 / mapped+graded_by 自洽 / **L1–L4 处置映射符合约定表** / 完整消息不被截断 |
| E 状态恢复（§1.5） | 6 | `device_state` 四态 / `recover_device` 返回 dict / **契约五键齐全** / recovered 语义=设备当前可用 / `hybrid` / `real` 如实响应 |
| F 硬纪律（§3） | 1 | **错误隔离分层**：芯片级→`device_recovery`、API 级→非 `device_recovery` |
| G 元信息 | 3 | `supports` 与 `info()["supports"]` 一致 / 键集合==`_CAPABILITY_KEYS` / `known_issues` 结构 |
| **H 内存句柄与生命周期（§1.6，2026-10-08 新增）** | 8 | `allocate` 五键句柄且**不含厂商指针** / 未声明 ⇒ 契约级拒绝 / 非正整数 ⇒ `ValueError` / **二次释放**与**陌生句柄** ⇒ `ValueError` / `memory_handle_count` 联动 / `memory_stats` 的 `allocated_mb` **取不到必须缺席**（不得填 0 冒充） / `record_stream` 未声明 ⇒ 保守同步**不抛错**且退化 +1 / `.native` 与退化**分开计数** |
| **I 设备上下文（§1.7，2026-10-08 新增）** | 10 | 未声明 `context_lifecycle` ⇒ 契约级拒绝 / `context_set`·`context_destroy` **只接受本层句柄**（默认上下文受保护） / `context_count` **不含默认上下文** / 两类句柄**不得互误用** / `context_query` **固定 6 键**（未声明者也返回同 6 键 + 具体原因） / 取值域（`present` 不得 0/1） / 查询**只读无副作用** / `recover_device` 的 context 三键只增不改且 `context_recreated` 不得与 `detail` 自相矛盾 / ⭐ **绑定语义**：销毁上下文后其流在使用点**必须报错** |
| **J 契约不变式（§1.8，2026-10-08 新增）** | 4 | I1 诚实声明 / I2 禁止伪造 / I3 失效受管 / I4 降级可观测 —— **委托** `conformance/contract_invariants.py` 的 `check_i1..i4`（**同一份实现**） |
| **K 流优先级（§1.9，2026-10-08 新增）** | 8 | `stream_priority_range()` **形状必须是 2 元组或 None**（不得透传厂商三元组） / 未声明 ⇒ 如实 `None` / 未声明 `control` ⇒ `create_stream(priority=…)` **契约级拒绝** / 越界·非 int·bool ⇒ `ValueError` / 声明 `control` ⇒ **必须**同时声明 `readback` / ⭐ 声明 `control` ⇒ **回读 == 请求** / 回读类型严格 `int` 或 `None` / 单点区间等价放行 |
| **L 流所有权与释放（§1.10，2026-10-08 新增）** | 6 | 厂商拥有的流 ⇒ `release_stream` **no-op 返回 `False`**（不得越权销毁）且该流仍可用 / 幂等 / `Stream.release()` 同语义 / **`owns_stream` 与 `release_stream` 必须一致** / 已释放的流再使用 ⇒ `RuntimeError` / 返回值**严格 bool** |
| **M 统一 API 面（契约 §1 标题即「统一 API 承诺」）** | 3 | ⭐ 契约 §1.1–§1.10 列出的统一 API 名在 `runtime` 上**真的可调用**（**本轮依此发现台账第 30 条**） / 公开常量（句柄字段·状态取值域）非空 / `runtime.__all__` 无幽灵导出 |

---

## 三、结果（补做后）

| 芯片 | 后端 | 真机结果 | 说明 |
|---|---|---|---|
| 910C | `ascend` | **`DUTY_RESPONSE_PASS` 39 / 0 / 0** | 首轮 39/0/0；补做后**回归仍 39/0/0**（无退化） |
| P800 | `kunlun` | **`DUTY_RESPONSE_PASS` 36 / 0 / 3** | 首轮 34/2/3（E3 缺口）→ 补做后 36/0/3 |
| MLU590 | `cambricon` | **`DUTY_RESPONSE_PASS` 36 / 0 / 3**（**39 项旧口径 · 首测**；现行结论见下方 §三·补） | 09-28 网络恢复后补跑一遍即 PASS；3 项 SKIP 均为「如实不具备」（见 §三 说明 + §六） |

**P800 的 3 项 SKIP 都是"如实不具备"**，不是缺口：
- `C13 elapsed_time`：`torch.cuda` 事件需 `enable_timing=True` 创建（契约外可选能力）
- `D5` / `F1`：本后端**无数字错误码**（Python 层不可得）⇒ 未声明 `error_map`，分级走 `message_hint`，
  已由 conformance F1 覆盖

---

### 三·补（2026-10-08 · 扩口径后的三家结果）

| 芯片 | 后端 | 真机结果（**78 项口径**） | 非空转验证（新增判据逐条注入） |
|---|---|---|---|
| 910C | `ascend` | **`DUTY_RESPONSE_PASS` 73 / 0 / 5** | **34 抓到 / 5 不适用 / 0 未抓到** |
| P800 | `kunlun` | **`DUTY_RESPONSE_PASS` 67 / 0 / 11** | **31 / 8 / 0** |
| MLU590 | `cambricon` | **`DUTY_RESPONSE_PASS` 61 / 0 / 17（2026-10-08 补齐轮 m1）** | **25 抓到 / 14 不适用 / 0 未抓到** |

> **SKIP 的含义不变**（如实不具备，不是缺口）。5 项与 11 项 SKIP 的逐条原因见各实例的职责文档 §1.1。
> ⭐ **扩口径顺带修掉两处真缺陷**：契约 §1.7 的 `context_set` 与 §1.9 的 `stream_priority_range`
> **只有后端实现、统一面未导出** ⇒ 下游按契约写会 `AttributeError`（缺陷台账 **第 30 条**）。
> 这正是「**职责审计的范围与契约齐平**」这件事本身的价值：口径一扩，缺口就暴露了。
> ⚠️ **实现约束（判据设计）**：设备上下文的 create/destroy **必须在子进程里跑**
> （同进程内做过后，后续 torch 算子报 CANN 内部错误、第二次 `check_i3` 直接段错误 rc=139）
> ⇒ 审计已把 `I1/I4/I5/I10/J3` 隔离到子进程（`INVASIVE_SIDS`），与 `probe_bc_contract.py` 的 C 组同法。

## 四、发现并补做的 3 处缺口

### 4.1 `recover_device` 返回**缺 `state`**（kunlun / cambricon）—— 契约违反

- **契约**（接口约定 §1.5 补充）：统一返回 `{ordinal, mode, recovered, state, detail}` **五键**
- **实测**：`ascend` 五键齐全；**`kunlun` / `cambricon` 只回四键**（缺 `state`）
  —— P800 首轮 E3 判 FAIL：`缺字段 ['state'] / 实际 ['detail','mode','ordinal','recovered']`
- **影响**：下游（监控方向）按契约读 `state` 在这两家拿不到 ⇒ **同一份上层代码换芯片行为不同**
- **补做**：两家 `recover_device` 补 `state`（取 `device_state(ordinal)`，与 ascend 同款；
  异常时如实回落 `"unknown"`）

### 4.2 四态命名**文档与实现不一致**（`UNKNOWN` vs `DESTROYED`）

- **约定文档** §1.5 写：`AVAILABLE / DEGRADED / ISOLATED / UNKNOWN`
- **实现**（三家共用的 `conformance/device_state.py::DeviceState`）：`AVAILABLE / DEGRADED / ISOLATED / **DESTROYED**`
- 两家后端 docstring 也写 `destroyed` ⇒ **文档是孤例**，且**两者并存 20 天无任何判据发现**
- 语义核对：实现里**没有 `UNKNOWN` 的路径**（新设备默认 `AVAILABLE`、探针失败置 `ISOLATED`），
  且 `DESTROYED`（已销毁/优雅退出）语义更具体 ⇒ **以实现为准更正文档**
- **补做**：更正接口约定 §1.5 + 在**变更记录**登记（接口签名未变、行为无变化）

### 4.3 `sync_timeout` 键三家不齐 —— 下游按能力名判定会跨芯片不一致

- `_CAPABILITY_KEYS` 原状：`ascend` **13** 项（含历史键名 `sync_timeout`）、`kunlun` / `cambricon` **12** 项
- **影响**：下游若用 `supports("sync_timeout")` 判定"是否有界同步"，在 `ascend` 得 `True`、
  在另两家得 `False` —— 而三家实际上**都有有界同步**（规范键 `bounded_sync`）⇒ 判定结论反了
- **补做**：把 `sync_timeout` 明确为 `bounded_sync` 的**弃用别名** ——
  ① 基类新增 `_CAPABILITY_ALIASES`，`supports()` 归一到规范键；
  ② 三家 `_CAPABILITY_KEYS` 补齐该键 ⇒ **键集合一致**；
  ③ 删除 `kunlun` / `cambricon` 里与基类完全同款的 `supports()` 覆写（收敛到唯一实现）

---

## 五、防漂移判据（离线自检新增 2 条）

本次两处契约违反（4.1 缺键、4.2 命名漂移）之所以能潜伏，是因为**三套件都没有对应判据**。
已补上，并做**非空转验证**（证明它们能真的失败）：

| 新判据 | 非空转验证（注入缺陷） |
|---|---|
| `recover_device` 返回**契约五键**齐全 | 临时移除 kunlun 的一个 `state` 键 ⇒ 判据 **FAIL**（`缺 ['state']`）✓ |
| `DeviceState` 四态成员 == 接口约定声明 | 临时给枚举加 `UNKNOWN` 成员 ⇒ 判据 **FAIL**（列出实现与约定的差异）✓ |

离线自检判据数因此从 **35 / 39 / 39** 增至 **37 / 41 / 41**（ascend / kunlun / cambricon）。

---

## 六、未做与边界（如实登记）

1. ✅ **MLU590 真机审计已补跑完成**（09-28 网络恢复后）：
   **`DUTY_RESPONSE_PASS` 36 OK / 0 FAIL / 3 SKIP**（用卡 0；3 项 SKIP 均为「如实不具备」）。
   离线契约自检 **41/0/0**、跨后端对称性 **5/0**；三处补做**真机验证生效**
   （E3「契约五键」判 PASS ⇒ `recover_device` 的 `state` 补做确认可用）。
   过程留痕：同日早先两次尝试均 SSH 超时（**间歇性**），网络恢复后一次跑通。
   ⚠️ 该实例的 `C13 elapsed_time` 报 `CNRT error: failed to call the driver-api function`
   —— 属契约外的可选能力，本后端未声明，故记 SKIP（不是缺口）。
2. **`C13 elapsed_time`** 在 P800 为 SKIP：属契约外可选能力（约定 §1.3 未列），不计缺口。
3. **审计工具自身修过 3 处**（值得一提，因为"工具质量 = 结论可信度"）：
   - `discover(names=None)` 用法错 ⇒ 假 FAIL（应为默认调用）—— **我的脚本错，不是后端缺陷**
   - `D5` 期望值原为"从 translate 结果反推" ⇒ **同义反复的空转判据**（必然通过），
     已改为**从码表取硬期望**
   - 芯片级/API 级错误样例的消息格式不符合码表的码提取正则
     （只认 `ret=N` / `error code is N`）⇒ 假 FAIL；已统一用 `_err_text()`
   - 样例构造用 `str(枚举)` 而非 `.name`（历史 IntEnum 的 `str` 给**数字**）⇒ 样例全空、
     `D5`/`F1` 被**静默 SKIP**；已修，并把"声明了 `error_map` 却取不到样例"**改判 FAIL**
     （工具失败不得伪装成"能力不具备"）

---

## 七、复跑命令（换芯片只改 `DC_BACKEND` / `--backend`）

```bash
# 真机（三实例同构；910C 需先按《名额纪律》清出带卡容器）
python3 scripts/duty_response_audit.py --backend <ascend|kunlun|cambricon> \
        --out <probes>/duty_audit_<backend>_<date>.json

# 离线（无设备，随时可跑）—— 含本次新增的 2 条防漂移判据
python3 scripts/backend_offline_check.py --backend <backend>
python3 scripts/backend_offline_check.py --all      # 跨后端对称性
```

**结论**：910C / P800 两个实例**职责响应 39 项全部通过**；MLU590 代码层补做已完成、
真机待网络恢复补跑。三处补做**不改接口签名**，接口约定版本仍为 v0.1。
