# 契约不变式 I1–I4 落地报告（工作包 B1 · 2026-09-29）

> 作者：Kistich（hliu553）｜对应：`VERIFICATION_MANIFEST` §1 第 9 条 / §3 **G8**（原状态 ⬜ **待做** → ✅ **已关闭**）
> 定位：把「契约不变式」从**只有名字、靠人工检查**，变成**机器判据**；判据在**真机与离线两处共用同一套核心函数**。

---

## 0 一句话结论

四条不变式（I1 诚实声明 · I2 禁止伪造 · I3 失效受管 · I4 降级可观测）**定义已补齐并落地判据**，
**两实例真机 4/4 通过**：

| 判定 | 910C（ascend） | P800（kunlun） |
|---|---|---|
| 真机 `--cases contract_invariants` | **4/4 → `CONTRACT_INVARIANTS_PASS`** | **4/4 → `CONTRACT_INVARIANTS_PASS`** |
| 离线桩（同套核心函数） | **75 / 0 / 1 跳过** | **76 / 0 / 1 跳过**（本机；另见 cambricon 64/0/0） |
| 非空转验证 | **5 处注入，5 处被抓** | 同（同一套函数） |
| 兼容性回归 | `cases` 13/13 · `infer_cases` 6/6 **判据串未变** | `cases` 13/13 · `infer_cases` 6/6 **判据串未变** |

⭐ 过程中**发现并修复一处真缺陷**（台账第 18 条）：`ascend` 把**弃用别名** `sync_timeout` 当"在册能力"列出，
另两家不列 ⇒ 同一份下游代码在不同芯片上读到不同的在册集合。

---

## 1 为什么补：此前后者"只有名字"

`VERIFICATION_MANIFEST` §1 第 9 条与 §3 G8 早已登记这一缺口，但**四条一直只有名字、没有定义** ——
全仓 `grep "I1|I2|I3|I4"` 只命中两处引用，且两处都只解释 **I2**（`mapped=True` 与
`graded_by=message_hint_*` 自相矛盾）。 ⇒ 拿不到定义就无法写判据，于是长期停留在"人工检查"。
本次先补定义（契约 **§1.8**），再写判据。

---

## 2 定义（契约 §1.8 摘要）

| # | 不变式 | 一句话 | 来源条款 / 首例 |
|---|---|---|---|
| **I1** | 诚实声明 | 声明的能力必须真的可调用；**未声明的必须显式拒绝** | 契约「声明即承诺」；台账第 ② 条 |
| **I2** | 禁止伪造 | 可观测字段之间**不得自相矛盾**；不得把"无依据"呈现为**有依据** | 台账 §2.3；第 ③ 条 |
| **I3** | 失效受管 | 已失效对象（句柄/上下文及其流）**不得静默可用**，须在使用点如实报错 | 契约 §1.6/§1.7；工作包 C 绑定语义 |
| **I4** | 降级可观测 | 降级必须**计数可查**；`.native` 取用与退化**分开计数**（语义相反） | 修订建议 §3；工作包 B-3/B-4 |

**判定细则**（完整版见契约 §1.8）：I1 五条（含"全集外键必须为 False""别名不得进声明集"）·
I2 四条（含"取不到就必须缺席，不许填 0 冒充"）· I3 四类负向 ·
I4 三条（结构 + 单调 + 两计数分离 + 跨流保护不得静默通过）。

---

## 3 落地：两处使用**同一套核心函数**

| 位置 | 形态 | 用途 |
|---|---|---|
| `runtime/conformance/contract_invariants.py` | 4 个 `case_i*`（runner 按 `case_` 前缀发现） | **真机**验：厂商原语真实形态下四条是否成立 |
| `scripts/backend_offline_check.py` 第 `[10]` 段 | 按**文件路径**加载同一模块，调 `CHECKS` 跑 stub 后端 | **无设备**拦回归（这正是本项"不需卡"的原因） |

> **为什么必须共用**：复制一份就会漂移 —— 离线改紧了、真机没跟上（或反之），
> 结果"离线绿、真机红"却查不出是判据不一致还是实现不一致。故离线用 `importlib` 按路径加载**同一文件**。

**判据串自证**（本次新增）：runner 支持用例模块定义 `VERDICT_TAG`，据此拼出 `CONTRACT_INVARIANTS_PASS/FAIL`
（**不定义则保持原行为** ⇒ 对 `cases` / `infer_cases` 零影响，已实测两者仍为 `CONFORMANCE_PASS`）。
理由：各"腿"的判定串应能自证是哪一项（同族：`TRAIN_LEG_PASS` / `INFER_LEG_PASS` /
`ERROR_RECOVERY_LOOP_PASS` / `STREAM_SEMANTICS_PASS`）。

---

## 4 实测

### 4.1 真机（两实例 · 2026-09-29）

| 用例 | 910C（ascend，3 设备） | P800（kunlun，1 设备） |
|---|---|---|
| `i1_honest_declaration` | PASS（入口存在性覆盖 **13/13** 项能力） | PASS（**10/13**） |
| `i2_no_fabrication` | PASS（陌生消息 → `L3_EXECUTION`/`replay`，`mapped=False`，`graded_by='default'`） | 同 |
| `i3_invalidated_governed` | PASS（二次释放/跨种类误用/二次销毁 → `ValueError`；**销毁后使用其流 → `RuntimeError`**） | PASS（**A 二次释放 → `ValueError`**；上下文分支**不适用** —— P800 未声明 `context_lifecycle`） |
| `i4_degradation_observable` | PASS（`.native` 计数 `0→1` 且**退化保持 0**；声明 `record_stream` ⇒ 原生路径零退化） | 同 |
| **汇总** | **4/4 · `CONTRACT_INVARIANTS_PASS`** | **4/4 · `CONTRACT_INVARIANTS_PASS`** |
| 兼容性回归（改 runner 之后重跑） | `cases` 13/13 · `infer_cases` 6/6（判据串仍 `CONFORMANCE_PASS`） | 同 |

证据：`../../910C/probes/recheck_*_ascend_20260929_r5b.json` ·
`recheck_cases_ascend_20260929_r5b.json` 等；P800 见
`../../P800/probes/recheck_contract_invariants_kunlun_20260929_r5.{json,log}`。
910C 的同批全套回归见 §4.3。

### 4.2 离线桩（本机，同套核心函数）

| 后端 | 离线自检 | 判据数变化 | `[10]` I1/I2/I3/I4 |
|---|---|---|---|
| `ascend` | **75 / 0 / 1 跳过** | 70 → **75** | PASS / PASS / PASS / PASS |
| `kunlun` | **76 / 0 / 1 跳过** | 71 → **76** | PASS / PASS / PASS / PASS |
| `cambricon` | **64 / 0 / 0** | 59 → **64** | PASS / PASS / **不适用** / PASS |

I3 在 `cambricon` **如实标"不适用"** —— 未声明 `memory_alloc` 与 `context_lifecycle`
（**未验证 ≠ 已确认不具备**）。

### 4.3 910C 第 5 轮全套回归（r5，与 B1 同窗口 · 11 项全绿）

| 判定项 | r5 | 说明 |
|---|---|---|
| 离线契约自检 | **75 / 0 / 1** | 判据数 70 → 75（+5：四条不变式 + 模块可加载） |
| 跨后端对称性 `--all` | **5 / 0** | — |
| 组件冒烟 | **52 / 0** | — |
| conformance 13 + 推理 6 | **13/13 + 6/6** | 判据串未变（兼容性） |
| **契约不变式** | **4/4 `CONTRACT_INVARIANTS_PASS`** | 本项新增 |
| 职责响应审计（39 sub-part） | **39 / 0 / 0** | — |
| 错误注入→恢复闭环 | **5 / 0 / 0**（`ERROR_RECOVERY_LOOP_PASS`） | — |
| 工作包 A 功能等价性 | **6 / 6** | — |
| 多流语义 / 配额 | **8 / 8 · 3 / 3** | — |

> 破坏面：`ab80d87`（ascend `_ctx_query_raw`）+ B1（ascend 声明集）**只影响 ascend 一家的后端目录**
> ⇒ 按"按破坏面覆盖"，**只需 910C 跑全套**；P800 只跑受影响的 conformance 三步（见 §4.1）。

---

## 5 ⭐ 非空转验证：5 处注入，5 处被抓

| # | 注入（模拟的真实缺陷形态） | 目标判据 | 结果 |
|---|---|---|---|
| **D1** | 把弃用别名 `sync_timeout` **放回** `ascend` 声明集 | I1③ | ✅ FAIL：`别名 'sync_timeout' 出现在 capabilities` |
| **D2** | `mapped=True` 但 `graded_by='default'`（**无依据却声称确定分级**） | I2①② | ✅ FAIL：`陌生消息却 mapped=True` + `自相矛盾` |
| **D3** | 去掉「销毁上下文后使用其流」的使用点拦截 | I3 | ✅ FAIL（本层判据 + 既有判据**双处**报警） |
| **D4** | 让 `.native` 取用**污染**退化计数 | I4② | ✅ FAIL：`污染了退化计数: 2 → 3（应不变）` |
| **D5″** | 未声明 `memory_alloc` 却**静默给出句柄**（实现层没有，伪造可用） | I1⑤ | ✅ FAIL：`未声明能力被静默接受`；**kunlun 正确保持绿**（它声明了该能力 ⇒ 规则不适用，说明判据有区分度） |

---

## 6 ⭐ 顺带发现的真实缺陷（台账第 18 条）

**现象**：`ascend._capabilities` **字面包含弃用别名 `sync_timeout`**，`kunlun` / `cambricon` 不含
⇒ `info()["capabilities"]` 在 910C 上是 **18 项**（含一个**已弃用**键），另两家 13 / 10 项。

**为什么是缺陷**：`sync_timeout` 自 2026-09-28 起已统一为 `bounded_sync` 的**弃用别名**
（三家 `supports("sync_timeout")` 均为 `True`，兼容入口保留）。既然别名在**任何实例上**都可查，
那"某个实例把别名当成**在册能力**列出来"就不是能力差异，而是
**同一份下游代码在不同芯片上读到不同的在册集合** —— 属"键集合同源"家族（台账 ③）。

**修法**：从 `ascend` 的**声明集** `_capabilities` 中移除（**保留**在能力全集 `_CAPABILITY_KEYS`
与别名映射里 ⇒ `supports()` 与 `info()["supports"]` 行为不变，只有 `capabilities` 列表归一）。

**影响面复核**：`_capabilities` 的读取方只有 `info()` 与离线/对称性自检（`grep` 全仓确认）
⇒ **未改任何接口签名，行为变化仅限该列表**。

---

## 7 判据自身的两处修正（如实登记，避免"判据看着对其实在误报/漏报"）

| # | 问题 | 发现方式 | 修法 |
|---|---|---|---|
| ③ | 初版写成「`capabilities` == 所有 `supports(k) is True` 的键」⇒ 在**别名键**上必然误报（`supports("sync_timeout")=True` 但它按设计不在 `capabilities`） | 首次运行即在 kunlun/cambricon 报错 | 改为**别名感知**：别名取值随规范键，且**别名不得出现在 `capabilities`**。⭐ **正是这个误报顺带暴露了第 18 条真缺陷** |
| ⑤ | 初版把「**抛任何异常**」都当成"显式拒绝" ⇒ 注入"去掉声明守卫"后**未被抓到**（更底层仍抛 `NotImplementedError`）—— **假通过** | D5 注入未命中 | 收紧为**只认契约级 `NotImplementedError`**；其它异常类型判 FAIL（疑似真 bug 被当成拒绝） |

> ③ 的教训：**判据要跟着"契约的扩展/弃用机制"走**（与"精确键集在只增不改契约下必然误报"同族）。
> ⑤ 的教训：**负向判据必须要求契约级异常类型**（同族：工作包 B/C 曾把 6 条负向判据收紧为「必须是 `ValueError`」）。

**另有两条工具性教训**（本轮真踩）：
① 注入脚本的锚点截断会留下语法错误 ⇒ **注入后必须先 `py_compile` 断言**、且**要看子进程退出码**，
否则会把"崩了"读成"0 失败"；
② **远端命令被 ssh 层终止 ≠ 远端进程结束** —— 本次因此出现过**两个 runner 并发跑同一张卡**
（已清理并单次重跑）；且用 `grep` 管道会**块缓冲**看不到进度 ⇒ 改为逐用例写日志。

---

## 8 边界与未做（如实，不补零）

- **真机已覆盖 910C + P800 两实例**；⏳ **MLU590 未跑**（本轮范围外，其 B/C 探测仍挂账）——
  **不得**由两个实例外推到第三方。
- **能力未声明 ⇒ 判据标"不适用"**（计为通过但**显式注明**）：**不得**因此把未声明能力当作已验，
  也**不得**据此声称该芯片不具备。
- I2 只检查**字段自洽与取值域**，不判断分级**是否正确**（分级正确性由 conformance F1/F2 与错误闭环覆盖）。
- I4 只验证"降级可观测"，**不评判降级带来的性能损失**。
- 四条均为**可观测性**约束；厂商原语真实形态仍须上机实测（离线结论**不得**外推）。

---

## 9 一键复跑

```bash
cd prototype

# ① 离线（无设备）：第 [10] 段会跑同一套核心函数
python3 scripts/backend_offline_check.py --backend ascend      # 期望 75/0/1
python3 scripts/backend_offline_check.py --backend kunlun      # 期望 76/0/1
python3 scripts/backend_offline_check.py --backend cambricon   # 期望 64/0/0

# ② 真机（容器内）：4 条用例，判据串自证
DC_BACKEND=ascend python3 runtime/conformance/runner.py --backend ascend \
    --cases contract_invariants --out conformance_invariants_ascend.json
# 期望：=== 汇总: 4/4 通过 === / 结论: CONTRACT_INVARIANTS_PASS

# ③ 向后兼容自证：既有用例集判据串不应改变
python3 runtime/conformance/runner.py --backend ascend --cases cases        # CONFORMANCE_PASS
python3 runtime/conformance/runner.py --backend ascend --cases infer_cases  # CONFORMANCE_PASS

# ④ 定义与判据同源
grep -n "### 1.8" docs/INTERFACE_CONTRACT_DC_20260908.md
grep -n "^VERDICT_TAG\|^CHECKS" runtime/conformance/contract_invariants.py
```
