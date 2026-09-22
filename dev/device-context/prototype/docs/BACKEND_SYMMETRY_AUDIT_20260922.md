# 跨后端对称性审计（2026-09-22）

> 负责人：Kistich（hliu553）｜ 触发：第三实例（寒武纪 MLU590）接入后的收口阶段
> **2026-09-22 晚补记（第二轮）**：本节新增 **第 9、10 条**（910C 真机暴露）与 **工具/资产类问题**，
> 并把第 8 条一并纳入"少被自检过的后端必有缺陷"这条主线。
> **2026-09-22 深夜补记（第三轮）**：910C 网络恢复 + 并发名额释放后做完整复核，再新增
> **第 11–15 条**；其中**第 11 条是此前"910C 上 conformance 跑不通"的真因**（不是环境问题，是我方原型缺陷）；
> 第 15 条由**训练腿切 `torch_npu`** 时暴露。
> **性质**：**审计 / 复核记录**（不是规范、不是操作手册）。规范正文见 `INTERFACE_CONTRACT_DC_20260908.md`，
> 修订建议见 `INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`。
> **写法**：每条缺陷给"现象 / 复现路径 / 归属 / 处置 / 防回归判据 / 各实例验证状态"。
> 未在某实例上验证的，一律显式标注**未验证**。

---

## 0. 结论速览

| # | 缺陷 | 归属层 | 影响实例 | 处置 | 防回归判据 |
|---|---|---|---|---|---|
| **第 6** | 无码表后端"诚实降级"只降了 `graded_by`，`mapped` / `error_code` 未降 ⇒ **把保守推断冒充成"确定分级"** | 后端适配层（`kunlun` / `cambricon` 同款） | P800、MLU590 | 三字段一起降级 | 离线自检 + smoke 各 3 条（**含他厂码串入负向测试**） |
| **第 7** | `flagos` **声明了 `device_state` 能力却没有实现该方法**（能力撒谎） | 后端适配层（`flagos`） | **910C（第一实例）** | 补实现（复用共享四态机） | 离线自检 `device_state 可调用` 判据 |
| **第 8** | `flagos` 的 `info()["supports"]` 手写了**第二份键名清单**，与 `_capabilities` 对不上 ⇒ 已声明能力在 `info()` 里恒显 `False` | 后端适配层（`flagos`） | **910C** | 改为按 `_CAPABILITY_KEYS` 全集派生 | 离线自检 `info()['supports'] 键集合 == 能力全集` |
| **证据污染 ①** | 错误闭环脚本把 **昇腾错误码 `507046` 硬编码在文案里** ⇒ 在无厂商码的后端上产出**假证据** | 验证资产（`proto_error_recovery_loop.py`） | P800、MLU590 | 改为从异常本身如实提取 | 见 §2.1（属"证据卫生"，非实现缺陷） |
| **证据污染 ②** | 同一脚本对**无码表后端**仍把 L4 注入描述为"按码表触发" | 验证资产 | P800、MLU590 | 按后端分化注入与期望；无码表后端改为**诚实降级负向测试** | 记录内自带 `expectation` / `expect_matched` |
| **第 9** | **`acl.init()` 返回值未检查** ⇒ 把"ACL 初始化/参数错误"**冒充成"设备同步超时"** | 后端适配层（`ascend`） | **910C** | 检查 init/set_device 的 rc；**只有已知超时码才映射 `TimeoutError`**，其余按码表如实分级 | 离线自检用**可控 rc 的假 pyACL** 测 4 条（真超时→TimeoutError；107000→**不得**是 TimeoutError 且须为 L2） |
| **第 10** | 统一错误对象**跨层类型不一致**：`api` 层是纯 `@dataclass`（**不能 `raise`**），`conformance` 层是 `Exception` | 框架类型契约（`api/errors.py`） | 全部 | 让 `FlagosError` 继承 `Exception` | `raise fe` / `except FlagosError` 可用性 |
| **工具 A** | 离线自检**并不离线**：只替换了 `torch`，真机 `PYTHONPATH` 上的**真实厂商绑定（`acl`）会被后端 import 到**并访问硬件 ⇒ 真机上自检**直接 traceback** | 验证资产（`backend_offline_check.py`） | **910C** | 新增**真实厂商运行时阻断器** + 为 ascend 注入**假 pyACL**；并把"离线性"设为**判据** | `未以真实文件形式加载任何厂商运行时` + `_host_vendor_bindings()` 如实报告 |
| **工具 B** | **未预期异常会让整轮自检崩掉、连汇总都打不出**（两次踩到） | 验证资产 | 910C（`device_state`）、910C（真实 `acl`） | 入口统一兜底：判 1 条失败 + 打印末帧 + **照常输出汇总** | `_guarded()` 包装 |
| **第 11** | **`use()` ≠ "设备命名空间就绪"**：后端懒加载 + 容器关 `TORCH_DEVICE_BACKEND_AUTOLOAD` ⇒ 拼 `"npu:0"` 报 `Expected one of cpu, cuda, …` ⇒ **conformance 整轮 ABORT**（不是用例失败） | 框架路径（`conformance/runner.py` 的 warm-up） | **910C**（此前 conformance 跑不通的**真因**） | warm-up **之前**先经后端触碰一次设备（`backend.device_count()`） | warm-up 路径可跑（910C conformance **13/13 + 6/6**） |
| **第 12** | `acl.init()` 的 **`100002`＝`ACL_ERROR_REPEAT_INITIALIZE`（重复初始化）被当成失败** ⇒ pyACL 被误判不可用；`flagos` 侧连带把 `memory_stats` 降级为进程级（`total_mb=0`） | 后端适配层（`flagos` + `ascend`） | 910C | 接受 `rc ∈ {0, 100002}`；其余非零按码表分级 | `memory_stats` 返回 **`memory_scope='device'`** 且 `total_mb>0` |
| **第 13** | **torch_fl `Event.query()` 语义缺口**：流上有工作时**恒 False**、**`ev.synchronize()` 成功后仍 False**、空流时真时假 ⇒ `wait_host()` **假超时** | **厂商运行时**（torch_fl，非我方缺陷） | 910C | 登记 `known_issues: FLAGOS-EVENT-QUERY-SEMANTICS` + 全变体实测矩阵；`wait_host()` 的 `False` 须读作「**未确认完成**」 | smoke 该项仍如实 **FAIL**（**不掩盖**）；已推荐改用阻塞 `synchronize()` / 显式依赖 |
| **第 14** | **OOM 注入在 `memory_stats["total_mb"]=0` 时静默退化成 `torch.empty(0)`**：不报错、不占显存，**却仍记"已注入"** ⇒ 假证据 | 验证资产（`proto_error_recovery_loop.py`） | 910C（第 12 条的连带后果） | `total<=0` 时**回退默认估值并打印提示** | 错误闭环 `oom` 项由「未触发异常」恢复为 `L1_RESOURCE` |
| **第 15** | **`init_process_group("hccl")` 早于 `use()`** ⇒ `hccl` 后端未注册 ⇒ `AssertionError: Unknown backend type hccl`（**厂商集合通信后端名也要先加载扩展才注册**） | 验证资产（`proto_train_leg.py`） | 910C（**训练腿切 torch_npu** 时暴露） | 初始化进程组**之前**先 `use()` + 触碰设备；并给 `ascend` 补默认 `DC_DIST_BT=hccl` | 训练腿 `TRAIN_LEG_PASS 6/6`（`dist=hccl`） |

**一句话（第二轮补充）**：第 9 条是**错误归因**里最坏的一种 —— 把"环境/初始化失败"报成
"设备同步超时"，下游会按 L3 去 `replay`，而正确动作是 L2 的 `raise`（**动作反了**）。
第 10 条则是**类型层面的雷**：名字叫"统一**错误**对象"却不能 `raise`。
工具 A/B 说明：**自检工具自身的质量问题会伪装成"后端没问题"**（跑不动 = 没结论，而"跑不动"很
容易被当成"不需要跑"）。

**一句话（第三轮补充）**：第 11/15 条是**同一根因的两种表现** ——
「**厂商扩展是懒加载的**」⇒ 凡是**要拼厂商专有字符串**（设备串 `"npu:0"`、集合通信后端名 `"hccl"`）的地方，
**都必须先经后端触碰一次设备**，否则框架根本不认识那个名字。
这类缺陷的表象极具误导性：`Expected one of cpu, cuda, …`、`Unknown backend type hccl`
**看着像"框架不支持/环境缺包"，实则是我方调用顺序问题**。
第 13 条则相反：那是**真的厂商缺陷**，唯一正确的处理是**如实登记并保留红灯**，不是改判据让它变绿。

**一句话**：第 6 条是**"看起来通过"的最危险形态**——记录里字段齐全、业务继续、五项闭环全绿，
只有交叉核对 `mapped` 与 `graded_by` 才发现两字段互相矛盾。第 7、8 条则说明
**"某家没被自检过"本身就是风险源**（工具此前只为 `cambricon` 内置 stub）。

---

## 1. 方法：这轮为什么能挖出来

1. **对称复跑**：在另两实例上用**同一份代码、同一套判据**重跑，差异即线索（第 5 个缺陷即由此暴露）。
2. **把"结果里的小字"当判据**：不再只看 `verdict`，而是逐字段核对
   （本次核心线索就是 `mapped=true` 与 `graded_by=message_hint_unexpected` 并存）。
3. **补齐自检的可达性**：把"无设备自检"从**一家可用**扩到**四家可用**（否则没被跑过的那家永远不暴露）。
4. **判据要能"真的失败"**：每条新判据都做**非空转验证**（见 §3），避免加了一批恒真的检查。

---

## 2. 逐条缺陷

### 2.1 第 6 条（最重要）：诚实降级三字段不一致 ⇒ 假"确定分级"

**现象**（MLU590 真机，错误闭环 `l4_by_code` 一条）：

```json
{ "inject": "l4_by_code(device_recovery 路径)",
  "raw": "AICORE exception, error code is 507015",
  "category": "L4_FATAL",
  "mapped": true,                        ← 契约含义：**确定分级**
  "graded_by": "message_hint_unexpected" ← 却明说"不是码表命中"
}
```

**机制**：共享翻译器 `conformance/errors.py` 的码表 `ACL_ERR_TO_CATEGORY` 是**昇腾 ACL 码表**，
而错误码**抽取规则却是通用的**（`ret=<数字>` / `error code is <数字>`）：

```python
m = re.search(r"ret\s*=\s*(\d+)", msg)
if not m:
    m = re.search(r"error code is\s*(\d+)", msg)
```

⇒ 只要异常消息里**恰好出现一个码表内的数字**（本例 `507015` 是昇腾的设备级错误码），
翻译器就返回 `graded_by="code_map"` **且 `mapped=True`**。

后端当时的处置是**只把 `graded_by` 改名**：

```python
if graded_by == "code_map":
    graded_by = "message_hint_unexpected"     # ← 只改了这一个字段
...
mapped=bool(getattr(fe, "mapped", False)),    # ← 仍然是 True
```

**危害**：`mapped=True` 在契约里的含义是"错误码命中映射表（**确定分级**）"
（见 `conformance/errors.py` 的 `FlagosError` 字段说明与 `api/errors.py` 注释）。
下游只读 `mapped` 就会把 L4_FATAL / device_recovery 当成**有码表依据的定论**，
而它实际是"消息里碰巧出现了一个别家的数字" ⇒ 可能触发一次**不必要**的设备恢复。
这正是接口约定不变式 **I2（禁止伪造）** 要防的形态。

**归属**：后端适配层（不是共享翻译器 —— 它对昇腾是**正确的**；错在无码表后端照抄了它的输出）。
`kunlun`（P800）与 `cambricon`（MLU590）**同款写法**，两家都受影响。

**处置**：`graded_by` / `mapped` / `error_code` **三个字段一起降级**
（`error_code` 置 `None`：该数字不是本厂商的码，留在结构化字段里会被下游当本厂商码去查；
原始文本已由 `root_cause` 原样保留，F4 不受影响）。

**防回归判据**（三处，覆盖"无设备"与"真机"两个层次）：

| 位置 | 判据 |
|---|---|
| `scripts/backend_offline_check.py`（无设备） | 3 条：码表内码串入 ⇒ `graded_by≠code_map`、`mapped is False`、`error_code is None` |
| `runtime/smoke_runtime.py`（真机） | 同 3 条（对**未声明 `error_map`** 的后端生效） |
| `conformance/cases.py` `f1`（真机，既有） | 未声明 `error_map` ⇒ 要求 `mapped=False`（本次沿用，未改动） |

**验证状态**：

| 实例 | 修复前 | 修复后 | 证据 |
|---|---|---|---|
| MLU590（cambricon） | ❌ `mapped=true` | ✅ `mapped=false` / `error_code=null` / 期望核对 ✅ | `MLU590/probes/error_recovery_loop_cambricon_20260922.json` |
| P800（kunlun） | ❌ 同款 | ✅ `mapped=False` / `error_code=None` | `P800/probes/I_error_recovery_loop_kunlun_refix_20260922.json`、`P800/probes/I_refix_kunlun_20260922.log` |
| 910C（ascend / flagos） | — **不适用**：两家**声明了** `error_map`，命中自己的码表是**正确的** | — | 不需改，行为未变 |

### 2.2 第 7 条：`flagos` 声明 `device_state` 却无实现

**现象**（`backend_offline_check.py --backend flagos`）：

```
AttributeError: 'FlagosBackend' object has no attribute 'device_state'. Did you mean: 'device_type'?
```

而它的 `_capabilities` 里写着 `"device_state"`。

**归属**：**910C 训练腿所用后端（第一实例）** 的能力声明与实现不符 ——
与 2026-09-20 在 `kunlun` 上发现的那处**同类**；
当时对 `kunlun` 的处置口径是**"补实现，不是删声明"**（四态机是芯片无关的共享资产），本次沿用同一口径。

**为什么一直没被发现**：① `device_state` **不在** `BaseBackend` 的 13 个抽象方法里 ⇒
ABC 实例化时不会拦（实例化检查只能拦抽象方法）；② `smoke_runtime.py` 的 `device_state` 判据
只对**被选中的后端**跑（910C 上该节选中的是 `ascend`），`flagos` 从未被这一节覆盖；
③ 离线自检工具此前**只为 `cambricon` 内置 stub**，`flagos` 跑不了。

**处置**：复用共享的 `conformance/device_state.py` 补上实现。
⚠️ **必须用标准 `import`（共享 `sys.modules`）**，不能像本类的 `_load_errors()` 那样用
`importlib` 独立模块名加载 —— `device_state` 是**有状态单例**，独立加载会得到两份状态机
（上层设置的状态后端查不到）。这条纪律来自 2026-09-20 的经验，已在代码注释中写死。

**同一次自检暴露的第二处（第 8 条）**：`info()["supports"]` 原先手写了一份键名清单
（`stream_priority` / `device_rebuild_real` / `device_rebuild_probe` / `error_code_map` / `bounded_sync`），
与 `_capabilities` 的键名（`recovery_real` / `recovery_probe` / `error_map` / …）**对不上**：

| | 后果 |
|---|---|
| 缺 10 键 | 已声明的 `recovery_probe` / `error_map` / `device_state` 等在 `info()` 里**恒显 False** |
| 多 3 键 | `device_rebuild_probe` / `device_rebuild_real` / `error_code_map` **根本不存在**于能力命名空间 |

⇒ 读者从 `info()` 会得出**完全相反**的结论。已改为与 `kunlun` / `cambricon` **同款**：
按 `_CAPABILITY_KEYS`（能力全集，12 项）逐项 True/False 呈现，清单不再是手写的第二份真相。

**验证状态**：

| 实例 | 离线自检（无设备） | 真机 smoke / conformance | 结论 |
|---|---|---|---|
| MLU590（cambricon） | ✅ 38/0 | ✅ smoke 46/0、13/13 + 6/6 | 无回归 |
| P800（kunlun） | ✅ 38/0/1 跳过 | ✅ smoke **46/0**、错误闭环 5/0/0 | 无回归 |
| **910C（flagos）** | ✅ 离线自检 31/0/2 跳过（本机跑） | ⛔ **未验证 —— 2026-09-22 当天 910C 主机 SSH 不可达**（`connect to host 10.120.72.27 port 22: Operation timed out`） | **待网络恢复后补真机验证** |

> ⚠️ **待办（不得省略）**：`flagos` 的修复**尚未在 910C 真机上跑过一次**。
> 网络恢复后须补：`smoke_runtime.py`（含 `device_state` 与 3 条新判据）+ `proto_error_recovery_loop.py`。
> 在补验之前，`flagos` 的 `device_state` 结论一律标注为**"代码层已修、真机未验证"**。

### 2.3 两处证据污染（非实现缺陷，但会直接产出错结论）

**① 硬编码昇腾错误码**：`proto_error_recovery_loop.py` 的超时注入原先把
`"进程内捕获 TimeoutError（真实 507046），进程存活"` 直接写在文案里。
`507046` 是**昇腾**错误码；在无厂商码的后端上该码**根本不会产生**（它们只有"超时上报"语义）
⇒ 这条记录是**假证据**，且会被后续读者当成"这家也能报出 507046"。
已改为**从异常本身如实提取**：有码就带出，无码就明说
（MLU590 实测输出：`异常消息内无厂商码（本后端为「超时上报」语义，不产生设备侧数字码）`）。

**② L4 注入的描述对无码表后端是错的**：原本对所有后端统一标为 `l4_by_code(device_recovery 路径)`，
暗示"按本厂商码表触发"。已按后端分化：

| 后端 | 注入消息 | 期望（写进记录的 `expectation`） |
|---|---|---|
| 声明 `error_map`（ascend / flagos） | **自己的**码样例（`SAMPLE_CODED_ERROR`） | `mapped=True` / `graded_by=code_map` |
| **未声明**（kunlun / cambricon） | 携带**共享码表内数字码**的消息 | `mapped=False` / `error_code=None` —— 即**不得**把他厂数字码当成本厂商码表命中 |

并且期望**参与判定**：记录里新增 `expectation` / `expect_matched`，
不符即判该条失败 —— 不能只看"业务还在跑"就放过（那正是"看起来通过"的来源）。

---

### 2.4 第 9 条：`acl.init()` 未检查返回值 ⇒ "ACL 错误"被冒充成"同步超时"

**现象链（910C 真机，2026-09-22）**：

```text
宿主带卡容器并发名额用尽
  → acl.init() 返回            500000 = ACL_ERROR_INTERNAL_ERROR   （CANN acl/acl_base_rt.h）
  → acl.rt.set_device(0) 返回  107002 = ACL_ERROR_RT_CONTEXT_NULL  （rt_error_codes.h）
  → synchronize_device_with_timeout 返回 107000 = ACL_ERROR_RT_PARAM_INVALID
  → 后端 raise TimeoutError("设备 0 同步超时（0ms），pyACL rc=107000")
```

**两处代码缺陷**（都在 `backends/ascend/backend.py`）：

1. `acl` 属性**只看"有没有抛异常"、不看 `acl.init()` 的返回值** ⇒
   把一个**未初始化的句柄**当成可用（`import acl` 成功即认为可用）。
2. 有界同步把**任何非零 rc** 都映射为 `TimeoutError` ⇒ 一个 `PARAM_INVALID`（L2，应为 `raise`）
   被报成**超时**（L3，处置是 `replay`）——**下游动作反了**。

**危害**：① 归因错（环境问题→设备/执行问题）；② 处置错（该上抛却去重放）；
③ 这条错误链**会掩盖真正的宿主约束**，让排查从"名额满了"跑到"同步为什么超时"。

**处置**：
- `acl` 属性检查 `acl.init()` 与 `get_device_count()` 的返回码，**不可用即降级为普通同步**
  并把原因留在 `acl_unavailable_reason`（供上层与证据读取）；
- 新增 `_ACL_SYNC_TIMEOUT_CODES = {107019, 107020, 507046, 507047}` ——
  **只有这些码**映射为 `TimeoutError`；其余非零码经 `translate_error` **按码表如实分级**；
- `synchronize` 里 `set_device` 的 rc **也**检查（它是这条链的中间环）；
- 错误码表增补 **`500000 = ACL_ERROR_INTERNAL_ERROR`**（来源是 `acl_base_rt.h`，
  **不是** `rt_error_codes.h` ⇒ 已在表内注明本表**跨两个头文件**、新增条目必须标出处）；
- 为 `ascend` 登记第一条 `known_issues`（**宿主资源约束**，非厂商缺陷；含"`npu-smi` 报 Health OK
  不等于名额有空"这条容易误判的提示）。

**验证状态**：

| 实例 | 结果 | 证据 |
|---|---|---|
| 910C | ✅ 真机复现整条链（`acl.init rc=500000`、`set_device rc=107002`、`sync rc=107000`） | 本次实测输出（见 §6 复现命令） |
| 910C / 全平台 | ✅ 修复后离线自检 **35/0/1**（含 4 条新判据，用**可控 rc 的假 pyACL** 驱动） | `backend_offline_check.py --backend ascend` |
| ⛔ 未做 | 修复后的**真机** smoke / 错误闭环**尚未跑**——910C 正被并发名额阻塞（见 §5） | — |

### 2.5 第 10 条：统一错误对象**跨层类型不一致**（`FlagosError` 不能 `raise`）

**现象**：同一个名字、两种类型：

| 层 | 文件 | 定义 | 能否 `raise` |
|---|---|---|---|
| API（对外统一） | `runtime/api/errors.py:69` | `@dataclass class FlagosError:` | ❌ **不能** |
| conformance（历史） | `runtime/conformance/errors.py:231` | `class FlagosError(Exception):` | ✅ 能 |

实测：`raise fe` → `TypeError: exceptions must derive from BaseException`。

**为什么现在才暴露**：既有用法**只把错误对象当值**（`isinstance` 判断 + 字段读取），
没有任何地方 `raise` 它。第 9 条的修复需要在后端里 `raise self.translate_error(...)`
—— **一写就炸**。这正是"修一个缺陷会暴露下一个"的典型。

**处置**：让 API 层 `FlagosError` 继承 `Exception`。
兼容性：现有用法（`isinstance` / 字段访问）**均不受影响**，dataclass 语义（`__init__`/`__eq__`）不变，
`__str__` 仍自定义。

**验证状态**：三实例均已在真机验证"**可 `raise`、可被 `except FlagosError` 捕获**"
（P800 已实测；MLU590 与 910C 侧待下次上机补跑，离线自检已覆盖）。

### 2.6 工具/资产类问题（不是后端缺陷，但同样致命）

**工具 A：离线自检其实并不离线。** stub 只替换 `torch`；真机上厂商绑定**本来就在 `PYTHONPATH` 上**
（910C 容器里 `acl` 可直接 import）⇒ `--backend ascend` 走到**真实 pyACL**、访问硬件、
抛 `TimeoutError`、自检**整个 traceback 结束**。三个后果：
① 自检在真机上用不了；② 结论被真实环境状态污染；③ "离线"这个前提是假的。

**处置**：新增 `_VendorRuntimeBlocker`（`sys.meta_path` 拦截 `acl`/`torch_npu`/… 的真实导入；
`sys.modules` 里我们注入的**假模块优先** ⇒ ascend 的假 pyACL 仍生效）；
把"**未以真实文件形式加载任何厂商运行时**"设为判据；并如实区分三种情形
（本机无绑定 / 有绑定但已被挡住 / 已被阻断器拦截）。

> ⚠️ 一条措辞教训：第一版提示语在"注入了假模块"时也印"**本机无厂商绑定**"——那是**错的**
> （910C 上真实 `acl` 明明存在）。已改为用 `PathFinder` 直接搜 `sys.path`（绕过 `sys.modules`）
> 来判断"本机到底有没有"，与"是否被挡住"分开报。

**工具 B：未预期异常让整轮自检无汇总地崩掉**（2026-09-22 踩到两次：`device_state` 未实现、
真实 `acl` 被 import）。这等于**把最重要的缺陷变成一次崩溃**。已加 `_guarded()` 入口兜底：
判 1 条失败 + 打印异常与末帧 + **照常输出汇总**。

**工具 C：把 SKIP 变成实测。** ascend 的有界同步原先因"stub 无法产生'任务未完成'状态"被 SKIP。
现在 stub 提供**可控 rc 的假 pyACL**，于是这条最关键的判据（rc 分类）**离线可测** ⇒
**第 9 条的缺陷本可以在上机前就被拦下**（实际是先上机才暴露 —— 这正是补上这条测试的价值）。

### 2.7 验证资产的口径更正：P800 的 S-7"5/5"不可复现（2026-09-22 复核）

按归档口径（910C 的 G1–G5）在 P800 上复跑时发现：

| 事实 | 说明 |
|---|---|
| P800 报告 §3.1 的"**按 910C 同口径 5/5**"**不成立** | 那 5 项（g1–g5：capture 基础 / 输入更新 / **多次 replay 稳定** / 显式流 / **捕获后其他流不受影响**）与 910C 脚本的 G1–G5（… / **捕获内切流** / 显式 stream）**不是同一组判据** |
| 该次运行的**脚本与 JSON 均未归档** | `P800/probes/` 下无对应文件 ⇒ 结论**不可复现**、判据无从核对 |
| 归档口径复跑结果 | **契约内 4/4 通过**（G1/G2/G3/G5，rel_err 全 0.00e+00）+ 1 项**契约外用法**不容忍 |
| 一处**根因不完整** | P800 §3.2 把首测失败归因于"捕获区内调 `synchronize`"；实测还有一类**无需同步**也失败的用法（捕获区内切流），且它**本就在上游契约之外**（正解是 `graph(g, stream=s)`） |

**这条的普遍意义**：**"未归档的结论"等于没有结论**。
只要脚本与结果没入库，别人（包括三个月后的自己）既不能复现也不能核对判据，
于是"5/5 通过"会被无差别地引用下去。已在 P800 报告中显式更正，并把判据口径固化进
`probe_graph_capture_stream_v2.py`（含"契约外用法降为宽容度观察项"这条处理）。

**另一处探针工程教训**：一次**失败**的捕获会弄脏捕获/分配器状态，
让**本可单独通过**的后续条目连带失败（实测：G4 失败后 G5 报
`Offset increment outside graph capture`，而 G5 单跑通过）。
⇒ 探针已改为"观察项放最后 + 条目失败后清理状态"，并用**逐项独立进程**复验过顺序无关性。

---

### 2.8 第 11–15 条（第三轮，910C 完整复核暴露）

#### 第 11 条：`use()` 之后设备命名空间并未就绪 ⇒ conformance **整轮 ABORT**

**现象**（910C，2026-09-22）：

```text
CONFORMANCE_ABORT: 后端 'ascend' 初始化失败: Expected one of cpu, cuda, ipu, xpu, ... :
device type at start of device string: npu
```

`flagos` 同样（`… device string: flagos`）。**注意形态**：不是某条用例失败，而是**后端初始化阶段就崩**
（`CONFORMANCE_ABORT`），所以看起来像"这个后端根本不可用"。

**根因链**（逐段实测）：

| 步 | 实测 |
|---|---|
| 容器环境 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0` ⇒ torch **不自动注册**厂商后端 |
| 裸 `import torch` | `hasattr(torch, "npu") == False` |
| `runtime.use("ascend")` 之后 | 仍为 `False` —— **`use()` 不触发厂商扩展导入**（后端是懒加载） |
| 调一次 `backend.device_count()` 之后 | 变为 `True` ✅ |
| `conformance/runner.py` 的 warm-up | `torch.zeros(1, device=f"{device}:0")` —— 恰好在**任何触发加载的调用之前** |

**为什么以前没暴露**：`kunlun` 的 `device_type` 是 `"cuda"`，属 torch **内置**命名空间，无需注册 ⇒
P800 一直正常。**这是"跨后端不对称"的又一例**：只有部分后端会踩到。

**处置**：`_setup_backend()` 在 warm-up 前插入 `_ = backend.device_count()`，
并把"**凡拼设备串前必须先经后端触碰一次设备**"写成纪律（记入手册）。

**验证**：910C conformance **13/13 + 6/6 全绿**（修复前为整轮 ABORT）。

#### 第 12 条：`100002` 是「重复初始化」，不是错误

`flagos` 的 `memory_stats` 最初降级为进程级，`memory_degraded_reason` 给出真因：

```text
'... total_mb': 0, 'memory_scope': 'process', 'memory_degraded_reason': 'acl.init rc=100002'
```

查 CANN 头文件确认语义（`acl/acl_base.h`）：

```c
static const int ACL_ERROR_NONE = 0;
static const int ACL_ERROR_REPEAT_INITIALIZE = 100002;
```

⇒ **torch_fl 已初始化过 ACL**，此处再 `init` 必然返回该码，**它不是失败**。
原先"非零即失败"的写法把 pyACL 判成不可用 ⇒ `memory_stats` 降级 ⇒ `total_mb=0`。
**同一处缺陷在 `ascend` 的 `acl` 属性也存在**（同类风险：把"可用"误判为"不可用"，
使有界同步静默降级、能力凭空丢失），已一并修正。

> **与第 9 条同源**：第 9 条是"非零 rc 一律当超时"，本条是"非零 rc 一律当失败" ——
> 共同纪律：**非零 rc 的语义必须逐码确认，不能一律套一个结论**。

**验证**：`memory_stats` 现返回 **`memory_scope='device'`**、`total_mb=62740` / `free_mb=62363` /
`used_mb=377`（设备级，与 `ascend`/`kunlun` 同口径），smoke 该项由 FAIL 转 PASS。

#### 第 13 条：torch_fl `Event.query()` 语义缺口（**厂商问题，如实保留红灯**）

**实测矩阵**（每个变体独立进程，见 `910C/probes/ev_matrix_20260922.log`、`ev_matrix2_20260922.log`）：

| 变体 | 结果 |
|---|---|
| 空流 / 默认流 上 record | `query()` 返回 `True`（**但 3 轮里有 1 轮全 False ⇒ 不稳定**） |
| **流上有工作**时 record | **恒 `False`（4/4）** |
| 流上有工作 + `torch.flagos.synchronize()` 后 | **仍恒 `False`** |
| 流上有工作 + **`ev.synchronize()` 成功返回后** | **仍 `False`（语义自相矛盾）** |
| 未 record 的事件 | `False` ✅（与契约一致） |

⇒ 该 `query()` **不能作为"是否完成"的判据**；依赖它的 `wait_host()` 会**假超时**。

**处置（刻意不"修绿"）**：登记进后端 `known_issues`（`FLAGOS-EVENT-QUERY-SEMANTICS`，severity=high，
含复现率、根因层、规避方式、上报对象、证据位置）；统一 `wait_host()` 的 `False` 必须读作
「**未确认完成**」而非「确认未完成」；建议等完成用阻塞 `synchronize()` 或`显式依赖`路径。
**smoke 该项保持 FAIL**（`42 通过 / 1 失败`）——**这是当前唯一未消除的红灯，且它是厂商缺陷**。

#### 第 14 条：OOM 注入在参数退化时**静默跳过**（假证据）

`inject_oom()` 用 `runtime.memory_stats(0)["total_mb"]` 算"3 倍显存"：

```python
total = int(stats.get("total_mb", 60000))   # 旧写法：只有"抛异常"时才回退默认值
n = int(total * 1024 * 1024 * 3)
return torch.empty(n, dtype=torch.uint8, device=dev)
```

第 12 条导致 `total_mb = 0`（**不抛异常**）⇒ `n = 0` ⇒ `torch.empty(0)`：
**既不报错也不占显存**，而错误闭环记录里仍写着"已注入" ⇒ 记录显示
`[oom(L1_RESOURCE 期望)] 未触发异常`，**看起来像后端缺陷，实为测试工具的证据污染**
（与"硬编码昇腾错误码"属同类问题）。

**处置**：`total <= 0` 时回退默认估值 **并打印提示**（提示中说明"若后端确实拿不到设备总量，
应在后端侧修，勿在工具里掩盖"）。

**验证**：修复后错误闭环 `oom` 项恢复为 `L1_RESOURCE / retry`，
910C 训练腿（`ascend`）错误闭环达 **5 闭环 / 0 跳过 / 0 失败**。

#### 第 15 条：`init_process_group("hccl")` 早于厂商扩展加载 ⇒ `Unknown backend type hccl`

**背景**：910C 训推统一 `torch_npu` 后首次跑训练腿即踩到：

```text
raise AssertionError(f"Unknown backend type {backend}")
AssertionError: Unknown backend type hccl
```

**根因**：`proto_train_leg.py` 里 `dist.init_process_group(DIST_BT)` 在第 124 行，
而触发 `torch_npu` 导入的 `runtime.use(BACKEND)` 在第 135 行 ——
**`hccl` 这个后端名要厂商扩展被 import 后才在 c10d 注册**，此时还不认识它。

⇒ **与第 11 条同根**：厂商扩展懒加载 ⇒ **凡是拼厂商专有字符串（设备串 / 集合通信后端名）之前，
都必须先经后端触碰一次设备**。已把这条写进脚本注释与手册。

**处置**：① `use(BACKEND)` + 触碰设备提前到 `init_process_group` 之前；
② 给 `ascend` 补默认 `DC_DIST_BT=hccl`（原本没有默认值，会落到 `gloo` ⇒ **静默退化为纯 CPU 集合通信**）。

**验证**：训练腿 `TRAIN_LEG_PASS 6/6`、`dist=hccl`，三类通信对照全对。

---

## 3. 判据非空转验证（新增判据必须能真的失败）

| 判据 | 非空转证据 |
|---|---|
| 第 6 条的 3 条离线判据 | 直接对**共享翻译器**投喂同一消息，实测返回 `mapped=True` / `graded_by=code_map` / `error_code=507015` ⇒ 若后端不做三字段降级，3 条必失败 |
| 第 8 条的键集合判据 | 用 flagos **修前**的键名清单对能力全集做集合差：**缺 10 / 多 3** ⇒ 判据必失败 |
| 第 9 条的 rc 分类判据 | 假 pyACL 注入 `rc=107000`：修前后端抛 `TimeoutError`（判据**必失败**），修后抛 L2 统一错误；`rc=507046` 仍须是 `TimeoutError`（防"一刀切改成不抛"） |
| 工具 A 的离线性判据 | 在 910C 上实测：修前 `acl` 被真实加载并让自检崩溃；修后判据通过且如实报出"本机存在真实绑定：['acl']" |

（两条都在本文件与提交信息里留了原始输出，便于复核。）

---

## 4. 工具侧改进（让"没被自检过"不再发生）

| 改进 | 说明 |
|---|---|
| **四家 stub** | 离线自检原先只为 `cambricon` 内置 stub，`--backend kunlun\|ascend\|flagos` **直接被拒** ⇒ 三家从未被自检过（第 7、8 条正因此长期未被发现）。现 stub 通用化为 `_vendor_stub(ns_name, vendor_modules, mem_mode)`，四家各一行注册 |
| **显式 SKIP 机制** | stub 是**语义空间**的：它无法模拟真实算子与厂商运行时。凡 stub 覆盖不到的判据**显式 SKIP 并写明原因**（如昇腾的有界同步走 acl 原语），既不误报 FAIL、也不混入"通过"计数 —— 汇总行改为 `X 通过 / Y 失败 / Z 跳过（stub 能力边界，非失败）` |
| **崩溃改为判 FAIL** | `device_state` 的调用原先会让整轮自检崩掉（只留 traceback、连汇总都打不出）—— 那正好把最重要的"声明与实现不符"变成一次崩溃。现捕获 `AttributeError` 并判 FAIL（第 7 条即由此暴露） |
| **新增键名漂移判据** | `info()['supports']` 的键集合必须 == 能力全集（有 `_CAPABILITY_KEYS` 用它，否则用 `_capabilities`）。smoke 既有的"取值一致"判据**覆盖不到键名漂移**（两边都取 False，取值一致性照样成立） |
| **真实厂商运行时阻断器**（第二轮） | 见 §2.6 工具 A：stub 只替换 `torch`，真机上 `acl` 等**会被真实 import** ⇒ 新增 `sys.meta_path` 阻断 + 假 pyACL + "离线性"判据 |
| **入口统一兜底**（第二轮） | 见 §2.6 工具 B：`_guarded()`，未预期异常判 1 条失败并**照常输出汇总** |
| **`--all` 跨后端对称性自检**（第二轮新增） | 把四家的"可观测形态"摆在一起：**硬判据 5 条**（必需方法齐备 / info 均提供 supports / 声明能力 ⊆ 能力全集 / 声明 `error_map` 必备码样例 / 全部可加载）+ **差异清单**（哪些能力只有谁声明、`device_type` 为何不同）。这是"统一 API"这句话的**可执行检验** |
| **SKIP → 实测**（第二轮） | ascend 的有界同步原先 SKIP，现由**可控 rc 的假 pyACL** 真测 ⇒ 第 9 条那类缺陷**离线即可拦下** |

**当前四家自检结果**（本机，无设备）：

| 后端 | 结果 |
|---|---|
| cambricon | **39 通过 / 0 失败 / 0 跳过** |
| kunlun | **39 通过 / 0 失败 / 1 跳过**（该家无独立可缺失的厂商扩展模块） |
| flagos | **32 通过 / 0 失败 / 2 跳过** |
| ascend | **35 通过 / 0 失败 / 1 跳过**（上一轮为 28/0/2：补 `info()` 后键集合判据生效、有界同步由 SKIP 变实测） |

**`--all` 对称性自检**：**5 通过 / 0 失败**（四家全部可加载；差异清单如实列出 6 项"仅某家声明"的能力、
`device_type` 四值不同（设计使然）、`supports` 键集合因 ascend 保留历史键名 `sync_timeout` 而略异）。

> 通过数不同是**结构性的**（各家声明的能力不同、按能力分支的判据条数不同），不是"谁更完整"。

---

## 5. 各实例验证状态（如实，含未完成项）

| 实例 | 离线自检 | `--all` 对称性 | 真机（smoke / conformance / 错误闭环 / 探针） | 备注 |
|---|---|---|---|---|
| **P800**（kunlun） | ✅ 39/0/1 跳过 | ✅ 5/0 | ✅ smoke **46/0** · conformance **13/13 + 6/6** · 错误闭环 **5/0/0** · 图捕获**契约内 4/4** · 配额 3/3 | 第 6 条复验 `mapped=False`/`error_code=None`；第 10 条 `raise`/`except` 可用 |
| **MLU590**（cambricon） | ✅ 39/0/0 | ✅（本机） | ⏳ **本轮真机复跑未做** —— 2026-09-22 晚间两台寒武纪主机 SSH 均超时（P800/910C 同刻正常，属该站点网络问题） | 判据口径变更后需重跑 smoke + 图捕获（旧 JSON 可**复算**为"契约内 4/4 + G4 容忍"，但新判据的**新鲜证据待补**） |
| **910C**（ascend） | ✅ **35/0/1**（容器内实跑） | ✅ **5/0** | ✅ **smoke 52/0** · **conformance 13/13 + 6/6** · **训练腿 6/6**（torch_npu + HCCL，2 卡/50 步）· **错误闭环 5/0/0** | **第 7/8/9/10/11/12/14/15 条均已真机复验**；第 13 条为**厂商缺陷**，如实保留红灯 |
| **910C**（flagos，**已转为备用**） | ✅ 32/0/2 | ✅ 5/0 | ✅ conformance 13/13 · 错误闭环 4/0/0 · smoke **42/1**（唯一 FAIL＝第 13 条厂商缺陷） | 2026-09-22 起**训练腿不再走本后端**（口径统一为 torch_npu），保留用于备用/历史复现 |

### 5.1 ✅ 910C"并发名额"阻塞已解除（2026-09-22）

> **结论**：清空宿主全部带卡容器后，`acl.init()` 立即返回 **0**、`get_device_count()` = **(16, 0)**、
> `set_device(0)` = 0 ⇒ **全部 16 chip 可用**。此前的卡点确为**他人容器占的名额**，与我们容器自身配置无关。
>
> **口径细化（新实测）**：该约束**按"挂载设备的容器数"计，与是否有活跃计算无关** ——
> 停容器前实测 16 个 chip **全部** `No process in device`（全场零计算）时 `acl.init()` 仍为 **500000**；
> 且**起容器本身不被拦**（已有 5 个 Up 时仍能起第 6 个），**只有设备 init 失败**。
> ⇒ 判定"能否上机"看 `docker ps` 里挂 davinci 的容器数，不要看 `npu-smi` 的进程列或 `Health`。

#### 原始记录（保留）



| 项 | 实测 |
|---|---|
| 现象 | 我们容器内 `acl.init()` 返回 **500000**（`ACL_ERROR_INTERNAL_ERROR`）；`get_device_count()` 返回 `(0, 507899 = DRV_INTERNAL_ERROR)`；控制台提示 `Different containers share the same device` |
| 芯片健康 | 全部 `Health: OK`（`npu-smi`）⇒ **健康 ≠ 名额有空** |
| 根因 | 宿主上**他人容器占了名额**：当前 5 个带卡容器 Up（`temp-cp-pcp2`、`temp-cp-dev`(owner `jliu171`)、`x-benchmark` 与 `flaggems-cann9.0.0`(owner `kangkai`)，外加我们的），而本方向记录的**并发上限是 3** |
| 已试 | 停止并重启我们自己的容器（排除自身残留会话）⇒ **同样失败**，确认与自身状态无关 |
| 需要谁 | 协调上述容器的使用者各停一个（或管理员介入） |

## 6. 复现命令

```bash
# ── 无设备（三台机器上都能跑；本轮已在 910C 容器内实跑）──
python3 prototype/scripts/backend_offline_check.py --backend <ascend|flagos|kunlun|cambricon>
python3 prototype/scripts/backend_offline_check.py --all          # 跨后端对称性自检

# ── 910C：复现"ACL 错误被冒充成同步超时"的整条链（容器内）──
python3 -c "import acl; print('acl.init rc =', acl.init())"      # 名额占满时 = 500000
#   再取 acl.rt.get_device_count() → (0, 507899)；acl.rt.set_device(0) → 107002

# ── P800：复验两条修复（需空闲卡）──
#   python3 -c 里 import runtime 后调 b.translate_error(RuntimeError("AICORE exception,
#   error code is 507015")) → 期望 mapped=False / error_code=None；再 raise 它应可被捕获

# ── 图捕获（契约内 4 项 + 1 观察项）──
CUDA_VISIBLE_DEVICES=2 DC_BACKEND=kunlun python3 prototype/probes/probe_graph_capture_stream_v2.py
```

> **一条流程纪律**：本节命令的结果都要**归档到对应芯片目录的 `probes/`**。
> §2.7 的教训就是"结论没归档 = 结论不可复现"。

## 7. 对接口约定 / 接入手册的启示（已提交修订建议）

1. **降级必须整组一致**：能力相关的可观测字段（`graded_by` / `mapped` / `error_code`）
   **要么一起降级，要么都不动**；"只改其中一个"会制造自相矛盾的记录。
   建议在接口约定正文里把它写成一条**不变式**，并明确"单一厂商码表被多后端共享"时的降级义务。
2. **`device_state` 的定位要明确**：它**不在** 13 个抽象方法里，因此 ABC 拦不住"声明了却不实现"。
   建议要么纳入必需抽象、要么在规范里写明它是可选能力**且声明即必须实现**，
   并把"声明的能力必须可调用"列进 conformance。
3. **`info()["supports"]` 的键集合按能力全集呈现**（不要手写第二份清单），
   建议写进接入手册的自检清单。
4. **验证资产的可达性与证据卫生同等重要**：
   "只有一家能跑的自检工具"= 其余三家的盲区；
   脚本里**硬编码某一家的错误码**= 在别家上直接产出假证据。两条都建议写进手册。
5. **错误码的"分级"必须先于"处置"**：只有**已知超时码**才可映射为超时语义；
   其他非零码必须按码表如实分级 —— 否则会出现"参数错误 ⇒ 让下游去 `replay`"这种**动作反了**的情形。
   建议写明："`TimeoutError` 仅用于真实超时；其余厂商 rc 一律走统一错误对象。"
6. **统一错误对象必须"可抛"**：同名类在不同层一个能 `raise`、一个不能，是典型隐藏雷
   （只在"第一次真的 `raise` 它"时炸）。建议规范里明确其基类，并加一条最小判据（`raise`/`except` 可用）。
7. **自检工具必须证明自己"离线"**：stub 只替换部分命名空间时，真机上的厂商绑定会漏进来。
   建议把"未加载真实厂商运行时"作为工具自身的**判据**，而不是靠假设。
8. **厂商扩展是"懒加载"，凡拼厂商专有字符串前必须先加载**（第 11、15 条）：
   设备串（`"npu:0"`）与**集合通信后端名**（`"hccl"` / `"flagcx"`）都只有厂商扩展被 import 后才被
   框架认识。建议在接口约定里明确：**"统一 API 的 `use()` 不保证厂商命名空间已注册；
   任何构造厂商专有字符串的调用点，必须先经后端触碰一次设备"**，并提供 `backend.device_count()`
   这类"最小触碰"约定。这条对**接入脚本/示例代码**尤其重要 —— 它们最容易直接拼字符串。
9. **厂商集合通信后端名必须有默认值**（第 15 条）：缺默认值时会落到 `gloo`，
   造成**静默退化为纯 CPU 集合通信**（训练照样跑完、loss 照样降，但设备侧通信没验到）。
   建议：**未探测过的芯片，宁可报错退出，也不给兜底**（本项目已对 `cambricon` 采用该做法）。
10. **厂商缺陷不可"改判据变绿"**（第 13 条）：遇到真实厂商缺陷时，正确做法是
    **登记 `known_issues` + 保留红灯 + 给出规避路径与上报对象**。
    把判据放宽（或把该项改判 SKIP）会让下游误以为能力可用 —— 与"降级必须整组一致"是同一原则的对偶面：
    **该降的要整组降，该红的要留着红。**
