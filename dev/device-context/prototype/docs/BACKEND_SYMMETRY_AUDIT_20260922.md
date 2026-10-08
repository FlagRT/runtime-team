# 跨后端对称性审计（2026-09-22）

> 负责人：Kistich（hliu553）｜ 触发：第三实例（寒武纪 MLU590）接入后的收口阶段
> **2026-09-22 晚补记（第二轮）**：本节新增 **第 9、10 条**（910C 真机暴露）与 **工具/资产类问题**，
> 并把第 8 条一并纳入"少被自检过的后端必有缺陷"这条主线。
> **2026-09-22 深夜补记（第三轮）**：910C 网络恢复 + 并发名额释放后做完整复核，再新增
> **第 11–15 条**；其中**第 11 条是此前"910C 上 conformance 跑不通"的真因**（不是环境问题，是我方原型缺陷）；
> 第 15 条由**训练腿切 `torch_npu`** 时暴露。
> **2026-09-29 补记（第四轮）**：以「分歧的业务代价」实验（工作包 A）为手段做**跨实例复验**，
> 新增 **第 16 条** —— 它不是"某个后端写错了"，而是**共享判据表本身"以个别厂商的文案为样本"**
> （规则是从昇腾/昆仑芯的写法反推的），于是同一故障在别家栈下的**等价说法**漏网。
> 三实例同口径复跑见 `../../MLU590/docs/CAMBRICON_MLU_REGRESS_AFTER_FIX_20260929.md`、
> `../../P800/docs/KUNLUN_P800_REGRESS_AFTER_FIX_20260929.md`。
> **性质**：**审计 / 复核记录**（不是规范、不是操作手册）。规范正文见 `INTERFACE_CONTRACT_DC_20260908.md`，
> 修订建议见 `INTERFACE_CONTRACT_REVISION_PROPOSAL_20260920.md`。
> **写法**：每条缺陷给"现象 / 复现路径 / 归属 / 处置 / 防回归判据 / 各实例验证状态"。
> 未在某实例上验证的，一律显式标注**未验证**。


> ⚠️ **路线 B（torch_fl）历史档案 —— 已冻结，非当前口径**
> 本文记录的是**当时**的做法与结论。路线 B 已于 2026-09-22 整体退出：原型里的该后端已**删除**，
> 三个芯片实例（昇腾 910C / 昆仑芯 P800 / 寒武纪 MLU590）**当前一律走厂商官方 torch 插件路线**
> （`torch_npu` / `torch.cuda` 兼容层 XPytorch / `torch_mlu`）。
> 当前口径见 `dev/device-context/README.md` 与各芯片目录 `README.md`；取舍依据见
> `summary/DEVICE_ABSTRACTION_ROUTE_AB_SUMMARY_20260922.md`；归档索引见
> `dev/device-context/prototype/docs/ROUTE_B_ARCHIVED_20260922.md`。

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
| **第 16** | **判据表「以个别厂商文案为样本」⇒ 参数类错误的等价说法漏网**：设备序号越界在寒武纪栈下原文是 `CNRT error: invalid argument.`，而 L2 规则只认 `invalid (device\|ordinal\|data\|op\|param)` ⇒ **无规则命中、兜底 `L3_EXECUTION`（`replay`）**，而契约期望 `L2_PARAM`（`raise`）⇒ **下游对一个永久性参数错误反复重放（动作反了）** | 共享判据表（`conformance/errors.py::_MESSAGE_HINTS`） | **MLU590**（P800/910C 文案恰好命中 ⇒ 未暴露） | L2 规则改为**按等价类覆盖**（`invalid argument`/`invalid value`/`illegal …`）；**刻意不做** `invalid \w+` 宽匹配 | 离线自检 +2 条「参数类文案等价类」判据（用两家真机原文），**非空转验证**：回退规则 ⇒ 恰好该条 FAIL（44/1） |

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

### 2.9 第 16 条（第四轮，跨实例复验暴露）：判据表以"个别厂商文案"为样本

#### 第 16 条：参数类错误的「措辞等价类」漏网 ⇒ 参数错误被当成执行错误重放

**现象**（MLU590 真机，工作包 A 实验 S1「设备序号越界」）：

| 路径 | 类别 | 处置 |
|---|---|---|
| ① 统一层 `runtime` | **`L3_EXECUTION`** ❌ | `replay` |
| ② 直接调厂商原生 | `L2_PARAM` ✅ | `raise` |

厂商栈原文（真机抓取）：MLU590 `CNRT error: invalid argument.`；P800 `CUDA error: invalid device ordinal`；
910C 带数字码走 `code_map`。

**根因**：`conformance/errors.py::_MESSAGE_HINTS` 的 L2 规则原为

```python
(re.compile(r"(invalid (device|ordinal|data|op|param))", re.I), ErrorCategory.L2_PARAM)
```

—— 它是**从个别厂商的文案反推**出来的（昇腾 / 昆仑芯的 `invalid device` 写法）。
CNRT 说的是 `invalid argument`：**既不在这张名词表里，也不含任何 L3 关键词** ⇒ 落到最后一条**兜底 `L3_EXECUTION`**。

**归属**：**我方**（共享判据表）。

**危害**：`L3_EXECUTION` 的契约处置是 `replay`。设备序号越界是**永久性**参数错误，
重放多少次都以同样方式失败 ⇒ **下游对着不可能成功的操作反复重放**；
与第 6 条（外来码表把参数类错误升级成 `L4_FATAL`、误触发设备级重建）是**同一危害家族的两个方向**：
**分级错了，下游动作就反。**

**处置**：L2 规则从"按名词枚举"改为"按**等价类**覆盖"：

```python
(re.compile(r"(invalid (device|ordinal|data|op|param|arg(?:ument)?s?|value|index)"
            r"|illegal (?:arg(?:ument)?s?|value|param|device))", re.I), ErrorCategory.L2_PARAM)
```

`invalid argument` / `invalid value` 是 **EINVAL 类调用方错误**的通用措辞 ⇒ 必须覆盖；
**刻意不做** `invalid \w+` 宽匹配（否则可能把 `invalid context` 一类 L4 场景吞进 L2）。

**防回归判据**：离线自检新增 **2 条**「参数类文案等价类 ⇒ L2_PARAM」，
输入用**两家真机抓到的原文**（不发明文案），**只碰文本、不碰设备** ⇒ 属于"上机前就能拦住"的场合。

**非空转验证**：把规则回退为旧版 ⇒ **恰好该条 FAIL**（`44 通过 / 1 失败`），
另一条（P800 措辞）仍 PASS；恢复后 `45/0`。

**各实例验证状态**

| 实例 | 状态 |
|---|---|
| MLU590（cambricon） | ✅ **真机复验通过**：等价性 **5/6 → 6/6**，`S1` 两路径均 `L2_PARAM`；离线自检 43→**45/0/0** |
| P800（kunlun） | ✅ 真机复验：离线自检 43→**45/0/1**；`S1` 本就正确（属**防回归**） |
| 910C（ascend） | 🟨 无设备判据 **40/0/1 已过**；**真机复跑未取得**（2026-09-29 11:19 起 SSH 超时，见当日日志）—— 属**未验证**，非"已确认无影响" |

> ⭐ **可推广的教训**：**判据表本身也要有"样本来源"意识** ——
> 凡规则是"照着某一家的报错文案写的"，就必须问一句：**同一故障在另两家的原文是什么？**
> 等价类没覆盖，规则再对也只是**在一家上对**。

---


### 2.10 第 17 条（第五轮，P800 上下文专项）：**「能力缺失」的根因表述本身是错的**

**现象**：工作包 C 首轮把 P800 未声明 `context_lifecycle` 的原因记为
「XPytorch 兼容层**未暴露**上下文原语，与 `recovery_real` 同因」。
该表述**未经实测到原语层**即写入 `WORKPACKAGE_BC_INTERFACE_20260929.md` 与 `P800/README.md`。

**实测更正**（2026-09-29，全文见 `../../P800/docs/KUNLUN_CONTEXT_SEMANTICS_20260929.md`）：

| | 原表述 | 实测 |
|---|---|---|
| 驱动层有无上下文 API | ~~无~~ | **有完整 `cuCtx*`（21 个）**，在 XPytorch 实际加载的 `libcuda.so.1` 里 |
| 能否创建 | —— | **能** —— `cuCtxCreate_v2` 返回真句柄，`push/pop current` 语义成立 |
| 谁在用 | —— | **XPytorch 自己在用**（torch 后 `cuCtxGetCurrent` **非 0**） |
| 真正的原因 | —— | **平台只允许一个上下文**（建 #2 → `rc=2`）+ **由框架自建** + **本层抢先去建会破坏框架** |

第三条硬证据：在 torch 之前建上下文 ⇒ torch 报 `CUDA error: invalid device ordinal`；
**销毁本层建的上下文 ⇒ torch 立即恢复 `512.0`**。

**危害**：把「**平台约束**」误记成「**厂商没做**」有两重代价 ——
① 误导后续接入者（以为此路不通而放弃本可补的能力）；
② 对外沟通失真（形同替厂商认领一个并不存在的缺口）。

**修法**：接口层新增 `context_query`（只读观测）把差异**写进字段**；
表述层逐处更正（`WORKPACKAGE_BC_INTERFACE_20260929.md` §11、`P800/README.md`、本台账、`STATUS.md`）。

**纪律（新增）**：**写「该栈未暴露 X」之前，必须实测到原语层** ——
**头文件入口表**（`grep` 全部 API 名）+ **库导出符号**（`nm -D`）+ **真调一次**（返回值/句柄/副作用），
三者缺一不可。同族：第 ⑤ 条（非零 rc 逐码确认）、第 ⑮ 条（厂商命名空间上的属性可能是"继承来的"）。

**补充（同日续修）**：本条自身也踩了同族的坑 —— 修正后的表述仍不够准确。
原写「**平台只允许一个上下文**（第二次 `cuCtxCreate_v2` 返回 `rc=2`）」，是**把机制简化成了现象**：
实测 `cuDevicePrimaryCtxGetState` 后可知，`rc=2` 的真实成因是 **primary context 已激活即占位**
（触碰前 `rc=3`/输出参数未写入 → 触碰后 `rc=0`、**`active=1`** → 此时显式创建被拒），
且该状态**有可观测判据**（`active` 字段）；910C 侧的 `get_primary_ctx_state=(1,0,0)` 与之同构。
⚠️ **代价**：简化表述不仅不准确，还**让一个可观测判据被忽略** —— 若当初就查 primary 状态，
「是否占位」本可成为判据而非推断。
**纪律**：写「平台只允许 N 个 X」这类**机制性断言**前，先找**厂商是否暴露了该机制的状态查询**；
有 ⇒ 用它做判据；没有 ⇒ 只能写现象（并注明是现象）。**现象与机制必须分清**。

### 2.11 第 18 条（第六轮，B1）：**弃用别名被当成「在册能力」列出**

**现象**（由契约不变式 I1 的判据在本机首次运行时报出）：
`ascend._capabilities` **字面包含** `sync_timeout`，而 `kunlun` / `cambricon` **不含**
⇒ `info()["capabilities"]` 在 910C 上是 **18 项**（其中一项是**已弃用**键），另两家 **13 / 10** 项。

**为什么是缺陷**：`sync_timeout` 自 2026-09-28 起已统一为 `bounded_sync` 的**弃用别名**
（三家 `supports("sync_timeout")` 均为 `True`，兼容入口保留）。
既然别名在**任何实例上**都可查，那"某个实例把别名当成**在册能力**列出来"就**不是真实能力差异**，
而是**同一份下游代码在不同芯片上读到不同的在册集合** —— 与第 ③ 条（键集合同源）同族：
可观测集合必须由**同一来源**派生。

**来源推断**：09-28 的别名统一**只改了三家的 `_CAPABILITY_KEYS`**（能力全集），
`ascend` 的**声明集** `_capabilities` 里的旧条目**未一并清理**（它是该键的原始拥有者）⇒ 残留。

**影响面复核**：`_capabilities` 的读取方只有 `info()` 与离线/对称性自检（全仓 `grep` 确认）
⇒ **未改任何接口签名**，行为变化仅限该列表。

**修法**：从 `ascend` 的 `_capabilities` 移除（**保留**在 `_CAPABILITY_KEYS` 与 `_CAPABILITY_ALIASES`
⇒ `supports()` 与 `info()["supports"]` 行为不变）。

**纪律（新增）**：**弃用 / 别名项一律不得进入「在册集合」类可观测字段**；
判据写法要**跟着契约的扩展与弃用机制走**（「精确键集」与「忽略别名」是同一族的两种误报）。

### 2.12 第 21 条（第七轮，A2 压测前置）：**「派生字段」用错了上游事实 ⇒ 未重建却声称已重建**

> 编号衔接：第 19 条（`SMOKE_TIMEOUT` 假失败）见《组内服务启动标准》v1.2；第 20 条（停机复查假信号）
> 见 `../../910C/docs/ASCEND_910C_LEGS_SERVE_RERUN_20260929.md` §4。二者均属**工具类**，未单列台账条目。

**现象**：`recover_device()` 的 `context_recreated` 由 `bool(mode == "real" and rec["recovered"])` **反推**。
实测三条误报路径（同一份返回里 `detail` 与它自相矛盾）：

| 场景 | `detail`（同一次返回） | `context_recreated`（修前） | 实际 |
|---|---|---|---|
| ascend：**健康设备**上调 `mode="real"` | `设备状态=available，无需重建，探活可用` | **`True`** | 未执行任何重建 |
| kunlun：`mode="real"`（`recovery_real` **未声明**） | `昆仑芯无设备级重置/重建原语 … → real 模式不支持` | **`True`** | 不可能重建 |
| cambricon：同上 | `寒武纪的设备级重置/重建原语未验证 ⇒ real 模式不支持` | **`True`** | 不可能重建 |

**为什么是缺陷**：契约 §1.5 把该字段的语义**定义**为「本次恢复**是否走了销毁并重建上下文的路径**」；
而 `recovered` 的语义是「**设备当前可用**」（契约 §补充，2026-09-09 明确定义）。
拿后者反推前者，是把**两件不同的事**当成一件 —— 违反 I1「诚实声明」与 I4「降级可观测」，
且是**同一 dict 内自相矛盾**（读者按 §1.5 读会得出"上下文已重建"的错误结论）。
这与第 ① 条同一思考方式：**凡"事后反推"，先问能否由源头直接给出该事实**。

**来源推断**：B/C 接口落地时把该键定为"**派生字段**"（`WORKPACKAGE_BC_INTERFACE_20260929.md` §风险表
自述"非厂商直供"），派生条件挑了手边最方便的两个字段，**没有回到"这件事的事实由谁掌握"**。

**影响面复核**：`context_recreated` 的读取方只有离线自检的**存在性**判据（全仓 `grep` 确认）
⇒ **未改任何接口签名**；修复后该字段的**取值**变化集中在"未重建"路径。

**修法**（源头给事实，不做反推）：
1. `conformance/recovery.py` 在**做决策的同一处**记录实际路径（`last_rebuild_path()`，
   `None` / `"probe"` / `"aclrtResetDevice"`）；
2. 各后端把该事实通过**私有键 `_rebuilt`** 回报（`ascend` 按 `last_rebuild_path()` 判断；
   `kunlun` / `cambricon` 从不重建 ⇒ 不回报即真话）；
3. `backends/base.py` 只认该事实：`context_recreated = bool(_rebuilt and mode in ("real", "hybrid"))`；
   **未回报视为未重建**（宁可不声明，不臆造）；
4. 顺带修 `ascend` 的 `detail` 文案：真实重建与探针重试成功**分开说**（原先一律写"重建成功"）。

**纪律（新增）**：**「派生字段」必须写明它的上游事实由谁掌握**；派生条件不得使用**近义但不同义**的字段
（`recovered` 是"能不能用"，不是"做没做"）。判据要查**取值域**而不只是字段名（见第 ⑫ 条）。

### 2.13 第 22 条（第七轮，A2 真机）：**字段时点未定义 ⇒ 同一 dict 内看着自相矛盾**

**现象**：真机隔离后真实重建成功时，`recover_device()` 返回
`{"state": "isolated", "recovered": True, "detail": "…真实重建成功…"}`。
契约 §1.5 只说「`state` 为设备四态之一，便于上层与监控方向判定」——**没说它是调用前还是调用后的状态**。
实现取的是**调用时（恢复前）**，但契约没写 ⇒ 下游按"当前状态"读会以为设备**仍处于隔离**。

**为什么是缺陷**：字段语义**缺时点定义** = 歧义；而本项目**禁止矛盾结论**，
一个"看起来自相矛盾"的返回值会在复核时被反复误读（本次即由 A2 压测的判定逻辑先撞上）。

**修法**（**只定口径、不改行为**）：契约 §补充 写明 `state` = **调用时（恢复前）**的状态，
并解释 `state="isolated"` + `recovered=True` 的**正确读法**（前者=从哪来，后者=现在能不能用），
恢复后状态请用 `device_state(ordinal)` 查询；同时在离线自检加判据**钉死时点**（防再漂移）。

**纪律（新增）**：**每个状态类字段都要问「哪个时点」**；定义不清楚时，先定口径再谈取值正确。

### 2.14 第 23 条（第七轮，A2 前置）：**共享资产被两条导入路径各加载一份 ⇒ 判据「静默空转」**

**现象**：`runtime/conformance/device_state.py` 既可作**扁平模块** `device_state` 导入
（后端 `_load_conformance()` 与本文档《用法》所用），又可作**包路径** `runtime.conformance.device_state`
导入。Python 让两条路径**各执行一次文件** ⇒ 进程内出现**两个状态机**。实测（无设备即可复现）：

```
flat is pkg                : False
flat._STATES is pkg._STATES: False
flat 视角 dev0 : isolated        # 经扁平名 set_device_state(ISOLATED)
pkg  视角 dev0 : available       # 包路径完全无感知
```

**为什么是缺陷**：一侧 `set_device_state()`（或 `register` 在途任务），另一侧**毫无感知**，而且**不报错**
⇒ 一切「先置状态（或先登记）、再判定」的判据都会**静默空转** —— 判的是**另一个世界**。
本次即因此产出一条**假 FAIL**（"state 时点漂移"），实测调用前=isolated 而返回 available，
追下去才发现"不是行为错，是**判据和被测对象不在同一个状态机上**"。
这与"一致性判据两处共用一套核心函数（按文件路径加载同一模块）"是**同一条纪律的反面**：
共用必须共用**同一个实例**，只共用**同一份文件**不够。

**来源推断**：`runtime/conformance/` 下**没有 `__init__.py`**（PEP 420 命名空间包），
于是同一文件天然有两条合法导入名；而共享资产带**可变模块级状态**（`_STATES`、`_INFLIGHT`、
`_LAST_REBUILD_PATH`），两份实例必然分叉。

**修法**：在带可变模块级状态的共享资产尾部把两条路径指向**先加载的那一份**
（`sys.modules.setdefault` 不覆盖已存在项）；并补导父包、挂好属性，
使 `import runtime.conformance.<name> as X` 的**语句形式**也走同一对象。
（`device_state.py` 与 `recovery.py` 各一处；`errors.py` 无模块级可变状态，未改。）

**纪律（新增）**：**共享资产同时存在两条合法导入名时，必须显式收敛为单一实例**；
并加一条判据把"单一实例"钉死（本次已入离线自检，属"不需卡"）。
诊断口诀：**判据失败时，先确认"判据与被测对象看的是不是同一个世界"**。

### 2.15 第 24 条（第八轮，(A) 方案落地）：**「构造函数收下了参数」≠「参数生效」**

**现象**：910C 上 `torch.npu.Stream(priority=7)` **接受**这个 kwarg 并返回一条流，
但用 C API `aclrtStreamGetPriority` **回读恒为 0**。也就是说：**Python 侧看不出任何异常**，
"设了优先级"与"没设"在调用方眼里**一模一样**。而同一栈的 pyACL
`aclrtCreateStreamWithConfig(priority)` 传 0/3/7 **全部可回读** ⇒ 参数是在**插件层**被丢掉的。

同一族还有一处更隐蔽的：`torch_npu._C._NPUStreamBase` 接受一个 `stream_ptr=` kwarg
（正是 stock torch `ExternalStream` 用的那个名字），**收下后静默忽略** ——
传进去的外部句柄既没报错、也没被用上，返回的是**流池里的另一条流**（12 个候选 kwarg 逐一实测，
只有 `stream_ptr` 被"接受且忽略"，其余一律 `TypeError: invalid keyword argument`）。

**为什么是缺陷**：这正是本层最贵的假象形态 —— 上层据此做调度决策，而设备侧**根本没这回事**；
而且它**不会被任何"命令跑通了"式验证发现**。

**修法**（本轮 (A) 方案）：
1. 能力键**拆两把钥匙**：`stream_priority_control`（能设置）与 `stream_priority_readback`（能回读），
   与既有的 `stream_priority`（能读范围）并列 —— 同 `device_state` / `device_state_control`、
   `context_query` / `context_lifecycle` 的拆法：**能读 ≠ 能改，能改 ≠ 能校验**。
2. `create_stream(priority=…)` 增**四道约束**（实现放基类唯一实现）：
   未声明 `stream_priority_control` ⇒ `NotImplementedError`（**显式拒绝，不静默降级**）；
   非 int / 越界 ⇒ `ValueError`；创建后**强制回读校验**，回读 ≠ 请求 ⇒ `RuntimeError`。
3. 三家**按实测如实声明**：MLU590 全 ✅（唯一能"设置+回读"的实例）；
   910C 与 P800 **只读**（范围可读、可回读，但**不声明设置**，理由各异：
   910C = 插件层缺入口，P800 = 优先级空间退化为单点）。

**关键认识**：**「声明」的粒度必须与「能验证的粒度」一致**。若只声明一个笼统的
`stream_priority`，下游无法从元信息判断"能不能真的设" —— 只能靠踩坑。

**非空转证据**：把 stub 的 `drop_priority` 打开（**精确复现 torch_npu 的静默丢弃行为**）⇒
"设置成功且回读一致"判据当场 FAIL（`RuntimeError: 流优先级未生效：请求 -3，设备回读 0`）。

### 2.16 第 25 条（第八轮）：**契约的「形状」也是契约 —— 而 stub 的"简化形态"会把它盖住**

**现象**：`stream_priority_range()` 的契约是 **2 元组** `(least, greatest)`，
但 `ascend` 实现**直接透传** pyACL `aclrtDeviceGetStreamPriorityRange` 的返回值 ——
真机是**三元组** `(7, 0, rc)`；而 `cambricon` 同接口返回 **2 元组** `(0, -3)`
⇒ **同一份下游代码在两台机器上读到不同形状**（`len()` 判断、解包都会分叉）。

**为什么长期没被发现**：离线 stub 里写的是 `lambda: (0, -1)` —— **2 元组**。
即 **stub 比真机"好看"**，把真实缺陷盖住了。这与 §2.6"stub 的不完整会伪装成实现的缺陷"
**正好相反**：这次是 **stub 的"简化"伪装成了实现的正确**。
⇒ 纪律：**stub 必须按真机形态写**（本轮已把 stub 改成返回三元组，判据当场可 FAIL）。

**修法**：`ascend` 归一为 2 元组；新增离线判据「`stream_priority_range()` 必须是 2 元组或 None」；
并把"形状是契约的一部分"写进基类文档。

**非空转证据**：把 `ascend` 改回透传三元组 ⇒ 判据 FAIL（`得到 (7, 0, 0)`）。

### 2.17 第 26 条（第八轮）：**隐式依赖调用方 CWD = 不可移植（"在一台机器上跑得通"可能是环境巧合）**

**现象**：`runtime/demos/demo_unified.py` 用 `Path(__file__).resolve().parents[1]` 当原型根 ——
但该文件在 `runtime/demos/` 下，`parents[1]` 是 **`runtime/` 目录本身**（正解是 `parents[2]`）。
同一句 `python3 runtime/demos/demo_unified.py`：
- **910C 容器内 ✅**：容器 `PYTHONPATH` 结尾多一个 `:` ⇒ **CWD 进了 `sys.path`**，靠调用方 CWD 恰好等于原型根蒙对；
- **P800 conda 环境 ❌**：`ModuleNotFoundError: No module named 'runtime'`。

**为什么是缺陷**：**可移植性层的示例脚本，自己不可移植**。更危险的是它的"通过"来自
**与代码无关的环境巧合** —— 换台机器就炸，而排查方向会被误导到"环境缺包"。

**修法**：改为 `parents[2]`，并支持 `DC_ROOT` 覆盖（与 `runtime/proto/*.py` 一致）。
**非空转证据**：修后从 `CWD=/` 且**不给 `DC_ROOT`** 运行 ⇒ 910C 与 P800 **双向通过**；
修前同条件在 910C 上同样会失败（已实测 `cd / && …` ⇒ `ModuleNotFoundError`）。


### 2.18 第 27 条（第九轮，(A) 真落地）：**`except AttributeError` 兜底过宽 ⇒「库没这个符号」与「代码有 bug」被混为一谈，且是静默的**

- **现象/证据**：kunlun `stream_priority_range()` 原写法是
  `try: fn = lib.xxx; fn.restype = ...; rc = fn(...) \n except AttributeError: return None`。
  当 `lib` 是**替身对象**（其同名属性是**绑定方法**，不允许赋值属性）时，`fn.restype = ...` 抛
  `AttributeError` ⇒ **被吞掉** ⇒ 返回 `None`。
  ⭐ 于是**现象**是「这台设备不支持读优先级范围」，**真因**却是「调用方式与替身不兼容」（是 bug）。
  本轮因为这个兜底**误判了一轮**（判据报出「声明了能力却没实现」的假 FAIL）。
- **修法**：用 **`getattr(lib, name, None)` 显式探针**判断「符号是否存在」；其余异常**一律传播**（不静默）。
  已用于 kunlun 四处：`cuCtxGetStreamPriorityRange` / `cuStreamGetPriority` / `cuStreamDestroy_v2` /
  `cuStreamCreateWithPriority`。
- **判据**（离线 `[8b-②]`，非空转验证）：注入「原语抛 `RuntimeError`」 ⇒ 必须**传播**（实测 PASS）。
- ⭐ **推广**：**兜底的范围必须等于「你真正想兜的那一件事」**。`except Exception` /
  `except AttributeError` 这类宽兜底会把「实现 bug」伪装成「能力缺失」——
  而能力缺失是**允许**的结论，于是它**永远不会被追查**。

### 2.19 第 28 条（第九轮，工具类）：**ctypes 取值必须用 `.value`；`int(c_int)` 会按「字节串」解析并抛 `ValueError`**

- **实测**（CPython **3.9 与 3.13 行为一致**）：

  | 表达式 | 结果 |
  |---|---|
  | `int(ctypes.c_int(2))` | ⛔ `ValueError: invalid literal for int() with base 10: b'\x02\x00\x00\x00'` |
  | `int(ctypes.c_uint(2))` | ⛔ 同上 |
  | `int(ctypes.c_void_p(5))` | ⛔ 同上（把内存当字节串解析） |
  | `ctypes.c_int(2).value` / `c_void_p(5).value` | ✅ `2` / `5` |

- **影响面**：只在**「替身 / 直调」**场景踩到（真机代码里 `int(handle)` 用的是 **Python `int`**，不触发）。
  本轮表现为：**假驱动自身崩**，且报错信息看起来像「实现有缺陷」。
- **修法**：统一 `_ctype_value(x) = x.value if hasattr(x, "value") else x`。
- ⭐ 与第 27 条同族：**这两条都在「替身/离线」这一侧把「工具问题」伪装成「被测对象问题」**。

### 2.20 第 29 条（第九轮，判据设计）：**造「可控假原语」要同构；判据要与「实际分支」对齐；detail 要与断言同源**

三小点，同一课 —— **判据自身的三类缺陷**：

1. **替身必须与真库「同构」**：厂商调用点是 `fn = lib.xxx; fn.restype = ...; fn.argtypes = ...`
   ⇒ 替身必须提供**可调用 + 可写属性**的对象（本轮写了 `_FakeFn`），**不能**用绑定方法
   （绑定方法不允许赋值属性 ⇒ 触发第 27 条那类静默吞掉）。
2. ⭐ **判据必须与实现的「实际分支」对齐**：我按「声明了 `stream_priority_control` ⇒ 一定走厂商 C API 建流」
   写判据；而**实际**在**单点区间**下基类走的是「**等价放行**」（厂商默认流 + 回读校验）
   ⇒ 判据断言错了分支 ⇒ 误报。
   修法：**按实际路径分流** —— 单点 ⇒ 等价放行（**不拥有** + `release` 为 no-op）；
   多档 ⇒ C API 路径（**本层拥有** + `release` 必须真销毁 + 释放后再用必须报错）。
3. **判据 `detail` 里的「实测值」必须来自被断言的**同一次**调用**：
   `release_stream` 是**幂等**的（第二次调用返回 `False`）⇒ 若 `detail` 里再调一次，
   打印出的「实测值」与判定依据**不是同一次** ⇒ **自相矛盾的证据**（本轮抓到两处）。
   修法：先存变量，**断言与 detail 同源**。

⭐ 一句话：**「造一个能骗过自己的替身」和「写一条能被自己的替身骗过的判据」是同一个错误的两种形态。**

### 2.21 第 30 条（第十轮，E1 口径对齐时发现）：**契约承诺的统一 API 名，统一面上根本不存在**

**现象**：契约第 1 章的标题是「**统一 API 承诺（下游直接使用）**」。其中
**§1.7** 的表格列了 `context_set(handle)`、**§1.9** 的表格列了 `stream_priority_range()`。
这两者都**只有后端实现**（`base.py::context_set`、各后端的 `stream_priority_range`），
而 `runtime` 模块**既没有定义、也没有导出**它们
⇒ 下游按契约写 `runtime.context_set(h)` / `runtime.stream_priority_range()` 会直接 **`AttributeError`**。

**为什么长期没被发现（这才是本条的重点）**：既有的入口存在性判据
（`runtime/conformance/contract_invariants.py` 的 **I1④**）查的是「**后端**入口」——
`callable(getattr(bk, e, None))`。`bk.context_set` **是存在的** ⇒ 判据通过。
于是「**后端有**」被当成了「**统一面有**」，而契约承诺的对象恰恰是**统一面**。
⇒ **两层被混为一谈：判据与被测对象不是同一个世界**（同族：§2.14 第 23 条）。

**取证**：对 `runtime/__init__.py` 做「契约承诺名 vs 顶层定义」静态比对 ⇒ 缺的恰好是这两个；
职责审计新增的 **M1** 判据（契约承诺名在 `runtime` 上真的可调用）**当场 FAIL**：

```
注入后：契约承诺但统一面取不到：['context_set'] ⇒ 下游按契约写 `runtime.<name>` 会 AttributeError
```

**处置（只增不改）**：`runtime/__init__.py` 补两个**纯转发**出口 + 两个 `__all__` 名字
（`context_set` / `stream_priority_range`，均为一行 `current().X(...)`）。

**防回归判据**：职责审计新增的 **M1 / M2 / M3** 三域 ——
契约承诺名可调用 · 公开常量（句柄字段/状态取值域）非空 · `runtime.__all__` 无幽灵导出；
三条都配了**逐条注入的非空转验证**（`probes/selfcheck_duty_audit_ext.py`），
910C **34 抓到 / 5 不适用 / 0 未抓到**、P800 **31 / 8 / 0**。

⭐ 一句话：**「声明即承诺」要问到底「向谁承诺」** —— 契约面向**统一面**，判据却守在后端面
⇒ 承诺在中间那一层掉在地上了。

### 2.22 第 31 条（第十一轮，MLU590 补齐时发现）：**委派出去的判据的「不适用」被压成了「通过」**

**现象**：同一份职责审计里**同一原因两种标签** ——

```
  [SKIP] H2/H3/H4/H5        未声明 memory_alloc ⇒ 该分支不适用
  [SKIP] I2/I3/I4/I5/I10    未声明 context_lifecycle ⇒ 该分支不适用
  [OK  ] J3                 不适用：本后端未声明 `memory_alloc` 与 `context_lifecycle`
```

**根因**：J 域把 `runtime/conformance/contract_invariants.py` 的 `check_i1..i4` **委托**过来
（刻意不重写：同一份实现才不会两处漂移）。被委托方在所需能力未声明时，按其模块 docstring 的
**既定约定**返回 `(True, "不适用：…")`（"判为通过但显式注明"）。而委托层把 `ok=True`
**一律**映成 `OK` ⇒ 一个**从未运行**的分支被记成「已响应」。

**为什么此前两家没暴露**：910C 两项能力都声明、P800 声明了 `memory_alloc`
⇒ 它们的 J3 **永远走真实分支**。**MLU590 是首个两项都不声明的实例。**

**双重危害**：① 读者会以为「61/0/17 = 全部职责已响应」，实际 J3 的分支从未运行 ——
这正是**第 30 条（E1 扩口径）要消除的那类误导**，而 E1 自己又造了一个同族；
② 非空转脚手架**无法**区分「判据失效」与「本机不适用」⇒ 误报「未抓到 1」（**假警报**，
会让人去修一个没坏的判据）。

**修法**：`contract_invariants.py` 把 `"不适用："` 提成**跨模块哨兵常量** `NOT_APPLICABLE`
（生产方与消费方**共用**，避免文案漂移导致审计侧静默退化成「通过」）；`duty_response_audit.py`
的 `_invariant()` 改为**三值翻译**（`True/False/None` ↔ `OK/FAIL/SKIP`）并显式识别哨兵。

**防回归判据**：非空转脚手架新增**无需设备**的 **§0 委派翻译层自检**
（`probes/selfcheck_duty_audit_ext.py`）—— 用受控替身直接喂四种返回形状并断言映射：

| 委派返回 | 期望审计状态 | 实得 |
|---|---|---|
| `True, "…"` | `OK` | ✅ |
| `None, "…"` | `SKIP` | ✅ |
| `True, NOT_APPLICABLE + "…"` | `SKIP` | ✅ |
| `False, "…"` | `FAIL` | ✅ |

⭐ **一句话**：**「判为通过但显式注明」是给「单条判据」看的，不是给「逐条点名的汇总」看的** ——
同一个信号，换一个消费口径就必须换一种表达；而**委托层是那个必须翻译的地方**。
另：**新分支必须证明它能真的报** —— `ok is None` 这条目前**没有任何生产方会返回**，
不做 §0 就只能是一个永不触发的 `if`。

**修后实测**：MLU590 `J3` → `[SKIP]`，审计 `61/0/17`，非空转 `未抓到 0`；
910C / P800 重跑**逐项不变**（`73/0/5` · `67/0/11`）⇒ 该修**不影响**声明了能力的实例。

### 2.23 第 32 条（第十一轮，MLU590 补齐时发现）：**「默认值」在跨实例时不可用，且失败信息指向错误方向**

三个**互不相同**的默认值缺陷，同一族（"默认值只在写它的那台机器上成立"）：

| # | 位置 | 原值 / 现象 | 为什么此前没暴露 |
|---|---|---|---|
| 1 | `runtime/demos/demo_unified.py` 的 `--backend` | 写死 `default="ascend"` 且**不读 `DC_BACKEND`**（其余 proto 脚本与全部探针都读）⇒ 在非昇腾实例上 `use("ascend")` → `info()` → `import torch_npu` ⇒ **`ModuleNotFoundError: No module named 'torch_npu'`**（**看着像"环境缺包"，实为"选错后端"**） | 该演示**只在 910C 上跑过** |
| 2 | `runtime/proto/proto_infer_serve.py` 的 `--backend` | 同上（同族、同修法） | 该脚本在 MLU590 上**从未跑过** |
| 3 | `scripts/serve_standard.sh` 的 **cambricon 分支** `MODEL` 默认值 | `MODEL=${MODEL:-/srv/hliu553/models/Qwen3-Embedding-0.6B}` 是**宿主路径**：容器内 `/srv/hliu553` 被挂成 `/work` ⇒ **容器内永不存在**；宿主上也**没有**这个目录（宿主真实位置 = `/srv/data/hf_cache/hub/…`）⇒ vLLM 抛 **`OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/srv/…'`**（**看着像"模型 id 格式不对"，实为"这个路径不存在"**） | 历史复现命令**总是显式传 `MODEL`** ⇒ 默认值走不到（该分支注释里恰好写着"首次在 MLU 容器内跑时请把实际报错回填"） |

**修法**：① ② 改为 `default=os.environ.get("DC_BACKEND", "ascend")`（与既有约定一致），
并给 `info()` 加兜底——把"厂商栈缺失"变成**可操作提示**；③ 改为**容器内可见**的 HF 快照路径
（`ls -d …/snapshots/*/`），并在 `esac` 之后补一个**共用前置断言**（路径为空或缺 `config.json`
⇒ 一条可操作的错误后 `exit 4`），同时覆盖 kunlun 分支的同类潜在问题（它给的是 HF **缓存根**
而非 `snapshots/<hash>/`，手册明确「给缓存根会报 `Unrecognized model`」）。

**防回归判据**：见 §3 表（`serve_standard.sh` 的前置断言；本轮 [A] 段**刻意不传 `MODEL`**
以真机验证新默认值，`vllm_serve.log` 的 `non-default args` 里 `model` = 容器内快照路径）。

⭐ **一句话**：**「默认值」是最容易撒谎的代码 —— 它只在写它的那台机器上被验证过**；
跨实例复用时，它既不报"我不适用"，还会把失败伪装成另一个方向的问题
（缺包 / 模型 id 格式错）。**默认值必须与它的运行环境同视角**（容器内路径），且**带存在性断言**。

### 2.24 第 33 条（第十二轮，MLU590 改走厂商 C API 时**当场发作**）：**登记表以 `id()` 为键 ⇒ id 复用让 `release_stream()` 越权销毁**

**缺陷**：`_owned_streams` / `_released_streams` / `_stream_ctx` 三张表都以 `id(obj)` 为键。
`id()` 只保证「同一对象的 id 在其**存活期内**不变」，**对象回收后 id 会被复用**
⇒ 一条**全新**的流可能撞上「已登记 / 已释放」那条的 id。三类后果（都属本项目禁止项家族）：

1. `release_stream()` **越权销毁厂商拥有的流** —— 契约 §1.10 规则 2 明令禁止，注释里写明
   **「比泄漏更糟」**（会让厂商栈内部状态崩坏）；
2. 一条**全新**的流被判「已释放」⇒ 使用点**误拦**；
3. 一条**全新**的流被判「绑定在已销毁的上下文上」⇒ 同样**误拦**。

**为什么长期没发作**：第 ① 类后果**只有当实例真的会产生「本层拥有」的流**时才有机会发生 ——
而三家在此之前**没有任何一家**在统一面上产出过这种流（910C 不声明 control、P800 单点等价放行、
MLU590 走 torch 侧）⇒ 触发条件从未满足。它此前只作为「**已知未修风险**」登记在册
（原话即「建议改 `weakref`」）。**第十二轮把 MLU590 改走厂商 C API 后，注册表第一次非空
⇒ MLU590 的 L1 当场 FAIL**：`release_stream(默认路径的流) 返回 True`（那条流是**厂商拥有**的）。

**修法**（`runtime/backends/base.py`）：登记时一并存**身份持有器**
（优先 `weakref.ref`；类型不支持弱引用时退化为**强引用** lambda —— 强引用同样保证身份可复核，
且保住对象 ⇒ 其 id 不会被复用）；查询 / 释放时**按对象身份复核**。
id 复用命中时**既不得销毁、也不得报 `True`**（返回值语义是「是否真的销毁了」）；
原条目**保留在册** —— 它记录的是「有一条本层拥有的流被回收却没释放」这个**泄漏事实**，不掩盖。

**防回归判据**：离线自检新增 **F 段**，**直接注入该情形**（把 a 的条目搬到 b 的 id 下，
等价于「b 拿到了 a 的 id」）：

| 断言 | 结果 |
|---|---|
| `owns_stream(b)` 必须**按身份**判定（不得把新流当成已登记的那条） | ✅ `False` |
| `release_stream(b)` **不得越权销毁**（必须如实返回 `False`） | ✅ `False` |

⭐ **一句话**：**`id()` 是「当前的地址」，不是「身份」** —— 凡是「以 id 为键、跨 GC 存活的登记表」，
都必须同时存一个**能证明身份**的东西（持有器），否则**回归一定会在某台机器上发生，只是时间问题**。
反过来的教训同样重要：**判据「长期没 FAIL」不等于「没问题」，可能只是触发条件没被满足**。

## 3. 判据非空转验证（新增判据必须能真的失败）

| 判据 | 非空转证据 |
|---|---|
| 第 6 条的 3 条离线判据 | 直接对**共享翻译器**投喂同一消息，实测返回 `mapped=True` / `graded_by=code_map` / `error_code=507015` ⇒ 若后端不做三字段降级，3 条必失败 |
| 第 8 条的键集合判据 | 用 flagos **修前**的键名清单对能力全集做集合差：**缺 10 / 多 3** ⇒ 判据必失败 |
| 第 9 条的 rc 分类判据 | 假 pyACL 注入 `rc=107000`：修前后端抛 `TimeoutError`（判据**必失败**），修后抛 L2 统一错误；`rc=507046` 仍须是 `TimeoutError`（防"一刀切改成不抛"） |
| 工具 A 的离线性判据 | 在 910C 上实测：修前 `acl` 被真实加载并让自检崩溃；修后判据通过且如实报出"本机存在真实绑定：['acl']" |
| 第 24 条的 4 条优先级判据（第八轮） | 5 处注入，**5/5 当场 FAIL**：① stub 开 `drop_priority`（**精确复现 torch_npu 静默丢弃**）⇒「设置+回读一致」FAIL；② 去掉能力门禁 ⇒「未声明必须显式拒绝」FAIL；③ `ascend` 改回透传三元组 ⇒ 形状判据 FAIL；④ 声明 `control` 却去掉 `readback` 声明 ⇒ 耦合判据 FAIL；⑤ 去掉越界校验 ⇒「越界 ⇒ ValueError」FAIL |
| 第 25 条的形状判据（第八轮） | 见上一行第 ③ 项（stub 改为按真机三元组返回后，该判据才具备"能 FAIL"的能力） |
| 第 26 条的 CWD 判据（第八轮） | `cd / && python3 <proto>/runtime/demos/demo_unified.py`：修前 910C/P800 均 `ModuleNotFoundError`，修后双向通过（**不给 `DC_ROOT`**，纯靠自身路径解析） |
| 第 30 条的 M1/M2/M3 判据（第十轮 · 职责审计扩口径） | **39 条逐条注入**：910C **34 抓到 · 5 本机不适用 · 0 未抓到**、P800 **31 · 8 · 0**（均 `SELFCHECK_DUTY_EXT_PASS`）；其中 M1 的注入就是「把 `context_set` 从统一面拿掉」⇒ 当场 FAIL（**正是本轮修的缺陷形态**） |
| 第 31 条的**委派翻译层**判据（第十一轮 · MLU590） | **§0 委派翻译层自检（无需设备）**：受控替身直接喂 4 种返回形状 ⇒ `True→OK` / `None→SKIP` / `NOT_APPLICABLE 哨兵→SKIP` / `False→FAIL` **全部符合**（910C / P800 / MLU590 三实例输出一致；MLU590 非空转 `未抓到 1 → 0`） |
| 第 33 条的 **id 复用**判据（第十二轮 · 离线 F 段） | **直接注入该情形**（把 a 的条目搬到 b 的 id 下）⇒ `owns_stream(b)=False`、`release_stream(b)=False`（两条均 PASS）。⚠️ **非空转验证本身也是新判据**：不做这一步就只是「看起来修了」 |
| 第 32 条的**默认值**判据（第十一轮 · MLU590） | ① `demo_unified.py` / `proto_infer_serve.py` 在同一实例上**不读 `--backend`**（走 `DC_BACKEND`）也能跑通（MLU590 `rc=0` / `SERVE_LEG_PASS 10/10`）；② `serve_standard.sh` 的 cambricon 默认 `MODEL` **刻意不传**也能就绪（`SERVE_STANDARD_PASS`，`vllm_serve.log` 的 `model` = 容器内快照路径）；③ 前置断言：路径为空/缺 `config.json` ⇒ `exit 4` 并打印三家的宿主/容器映射 |

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
| **MLU590**（cambricon） | ✅ **88/0/1**（2026-10-08 m1 轮） | ✅ **7/0** | ✅ **m1 轮 20 项全绿**：smoke **46/0** · conformance **13/13 + 6/6** · 契约不变式 **4/4** · 职责审计 **61/0/17（78 项）** · 错误闭环 **5/0/0** · B/C 探针 PASS · 优先级 API **7/7** · 释放/所有权 PASS · 训练腿 **6/6**（`cncl`）· 推理腿 **13/13** · 服务化 **PASS + `SERVE_LEG_PASS 10/10`** | 网络 2026-10-08 恢复后一次窗口跑完；判据口径变更后需重跑 smoke + 图捕获（旧 JSON 可**复算**为"契约内 4/4 + G4 容忍"，但新判据的**新鲜证据待补**） |
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
