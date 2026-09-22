# 跨后端对称性审计（2026-09-22）

> 负责人：Kistich（hliu553）｜ 触发：第三实例（寒武纪 MLU590）接入后的收口阶段
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

## 3. 判据非空转验证（新增判据必须能真的失败）

| 判据 | 非空转证据 |
|---|---|
| 第 6 条的 3 条离线判据 | 直接对**共享翻译器**投喂同一消息，实测返回 `mapped=True` / `graded_by=code_map` / `error_code=507015` ⇒ 若后端不做三字段降级，3 条必失败 |
| 第 8 条的键集合判据 | 用 flagos **修前**的键名清单对能力全集做集合差：**缺 10 / 多 3** ⇒ 判据必失败 |

（两条都在本文件与提交信息里留了原始输出，便于复核。）

---

## 4. 工具侧改进（让"没被自检过"不再发生）

| 改进 | 说明 |
|---|---|
| **四家 stub** | 离线自检原先只为 `cambricon` 内置 stub，`--backend kunlun\|ascend\|flagos` **直接被拒** ⇒ 三家从未被自检过（第 7、8 条正因此长期未被发现）。现 stub 通用化为 `_vendor_stub(ns_name, vendor_modules, mem_mode)`，四家各一行注册 |
| **显式 SKIP 机制** | stub 是**语义空间**的：它无法模拟真实算子与厂商运行时。凡 stub 覆盖不到的判据**显式 SKIP 并写明原因**（如昇腾的有界同步走 acl 原语），既不误报 FAIL、也不混入"通过"计数 —— 汇总行改为 `X 通过 / Y 失败 / Z 跳过（stub 能力边界，非失败）` |
| **崩溃改为判 FAIL** | `device_state` 的调用原先会让整轮自检崩掉（只留 traceback、连汇总都打不出）—— 那正好把最重要的"声明与实现不符"变成一次崩溃。现捕获 `AttributeError` 并判 FAIL（第 7 条即由此暴露） |
| **新增键名漂移判据** | `info()['supports']` 的键集合必须 == 能力全集（有 `_CAPABILITY_KEYS` 用它，否则用 `_capabilities`）。smoke 既有的"取值一致"判据**覆盖不到键名漂移**（两边都取 False，取值一致性照样成立） |

**当前四家自检结果**（本机，无设备）：

| 后端 | 结果 |
|---|---|
| cambricon | **38 通过 / 0 失败 / 0 跳过** |
| kunlun | **38 通过 / 0 失败 / 1 跳过**（该家无独立可缺失的厂商扩展模块） |
| flagos | **31 通过 / 0 失败 / 2 跳过** |
| ascend | **28 通过 / 0 失败 / 2 跳过**（探活与有界同步走厂商运行时，stub 覆盖不到） |

> 通过数不同是**结构性的**（各家声明的能力不同、按能力分支的判据条数不同），不是"谁更完整"。

---

## 5. 对接口约定 / 接入手册的启示（已提交修订建议）

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
