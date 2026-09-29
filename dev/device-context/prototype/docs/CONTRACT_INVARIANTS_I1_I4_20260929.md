# 契约不变式 I1–I4 落地报告（工作包 B1 · 2026-09-29）

> 作者：Kistich（hliu553）｜对应：`VERIFICATION_MANIFEST` §1 第 9 条 / §3 **G8**（原状态 ⬜ **待做**）
> 定位：把「契约不变式」从**只有名字、靠人工检查**，变成**机器判据**（真机 + 离线两处，同一套核心函数）。

---

## 0 一句话结论

四条不变式（I1 诚实声明 · I2 禁止伪造 · I3 失效受管 · I4 降级可观测）**定义已补齐并落地判据**：

- **真机**：`python3 runtime/conformance/runner.py --backend $B --cases contract_invariants` ⇒ `CONTRACT_INVARIANTS_PASS 4/4`
- **离线桩**：`python3 scripts/backend_offline_check.py --backend $B` 第 `[10]` 段（**无设备即可拦住回归**）
- **离线实测**：`ascend 75/0/1` · `kunlun 76/0/1` · `cambricon 64/0/0`（判据数 70→75 / 71→76 / 59→64）
- **非空转验证**：**5 处注入，5 处被抓**（含 2 处判据自身的缺陷修正）
- ⭐ **顺带发现并修复一处真缺陷**（台账第 18 条）：`ascend` 把**弃用别名** `sync_timeout` 当"在册能力"列出，另两家不列 ⇒ 同一份下游代码在不同芯片上读到不同的在册集合。

---

## 1 为什么补：此前后者"只有名字"

`VERIFICATION_MANIFEST` §1 第 9 条与 §3 G8 早已登记这一缺口，但**四条一直只有名字、没有定义** ——
全仓 `grep "I1|I2|I3|I4"` 只命中两处引用，且两处都只解释 **I2**（`mapped=True` 与
`graded_by=message_hint_*` 自相矛盾）。

⇒ 拿不到定义就无法写判据，于是长期停留在"人工检查"。本次先补定义（契约 **§1.8**），再写判据。

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

---

## 4 离线实测（本机，2026-09-29）

| 后端 | 离线自检 | 判据数变化 | `[10]` I1/I2/I3/I4 |
|---|---|---|---|
| `ascend` | **75 / 0 / 1 跳过** | 70 → **75** | PASS / PASS / PASS / PASS |
| `kunlun` | **76 / 0 / 1 跳过** | 71 → **76** | PASS / PASS / PASS / PASS |
| `cambricon` | **64 / 0 / 0** | 59 → **64** | PASS / PASS / **不适用** / PASS |

关键取证（`[10]` 段自报）：
- I2：陌生消息 → `L3_EXECUTION` / `replay`，`mapped=False`，`graded_by='default'` ✓
- I3（ascend）：二次释放 → `ValueError`、跨种类误用 → `ValueError`、二次销毁 → `ValueError`、
  **销毁后使用其流 → `RuntimeError`**、未登记句柄 → `ValueError`
- I3（cambricon）：**如实标"不适用"** —— 未声明 `memory_alloc` 与 `context_lifecycle`
  （**未验证 ≠ 已确认不具备**）
- I4：`.native` 计数 `0→1` 且**退化保持不变**；无原生 `record_stream` ⇒ 保守路径 + 退化 +1 ✓

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
> 另记一条**工具性**教训：注入脚本的锚点截断会留下语法错误 ⇒ 必须**注入后先 `py_compile` 断言**
> 且**看子进程退出码**，否则会把"崩了"读成"0 失败"（本次真踩了）。

---

## 8 边界与未做（如实，不补零）

- ⛔ **真机复跑未做**：`--cases contract_invariants` 尚**未在任何真机上执行**（本机无设备）。
  ⇒ 当前只能声明"**离线桩通过**"，**不得**当成真机结论。安排在与 **910C/P800 第 5 轮回归（r5）同一窗口**。
- **能力未声明 ⇒ 判据标"不适用"**（计为通过但**显式注明**）：**不得**因此把未声明能力当作已验，
  也**不得**据此声称该芯片不具备。
- I2 只检查**字段自洽与取值域**，不判断分级**是否正确**（分级正确性由 conformance F1/F2 与错误闭环覆盖）。
- I4 只验证"降级可观测"，**不评判降级带来的性能损失**。
- 四条均为**可观测性**约束；厂商原语真实形态仍须上机实测（离线结论**不得**外推）。

---

## 9 一键复跑

```bash
cd prototype

# ① 离线（无设备）：三条不变式会跑在 stub 后端上，第 [10] 段
python3 scripts/backend_offline_check.py --backend ascend      # 期望 75/0/1
python3 scripts/backend_offline_check.py --backend kunlun      # 期望 76/0/1
python3 scripts/backend_offline_check.py --backend cambricon   # 期望 64/0/0

# ② 真机（容器内）：4 条用例
DC_BACKEND=ascend python3 runtime/conformance/runner.py --backend ascend \
    --cases contract_invariants --out conformance_invariants_ascend.json
# 期望：CONTRACT_INVARIANTS_PASS 4/4

# ③ 交叉核对：定义与判据同源
grep -n "### 1.8" docs/INTERFACE_CONTRACT_DC_20260908.md
grep -n "^CHECKS" runtime/conformance/contract_invariants.py
```
