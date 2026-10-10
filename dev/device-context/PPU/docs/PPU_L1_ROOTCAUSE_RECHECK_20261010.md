# L1 根因复核（平头哥 PPU · 2026-10-10 第二轮）

> 起因：用户质疑「**可能你现在抓的问题依旧只是现象，而没找到根本原因**」。
> 本文是对同日《同口径自审》§4 与提交 `a6fade3`（「id 键表改强持有」）的**独立复核**。
> 复核方法：**不复用上一轮的观测方式** —— 先用代码逻辑与本机实验**证伪原根因**，
> 再做**交替对照**的真机实验（A/B/A/B，排除时间漂移），并让探针在命中登记表时输出现场。
>
> **三句话结论**：
> 1. ⚠️ **原「根因」解释不成立**（弱引用在对象已回收时 `holder()` 返回 `None`，身份复核**同样拦下**）；
> 2. ✅ **但该改动不是无效的**：在**可复现的探针序列**下，**弱引用 5/5 FAIL · 强持有 5/5 OK**（交替跑）；
> 3. 🔴 **真正的疑点已收敛到一处**：命中的登记条目里存的对象**就是那条默认流本身**（类型 `Stream`），
>    而代码上 `_register_owned_stream` 的唯一调用点只登记 `ExternalStream` ⇒ **存在一条尚未定位的「默认流被登记」路径**。
>    对外只能表述为「**L1 现象在可复现序列下已消除；根因未定位**」，**不得**说「已修复（根因＝持有器强度）」。
>
> 证据目录：`../probes/l1_fix_20261010_out/full/`（pre/post 完整 log+JSON）·
> `../probes/l1_fix_20261010_out/repeat/`（审计 15+15 次）· `../probes/l1_recheck_20261010/out/`（探针 2+10 次、登记来源探针）。

---

## 1. 被复核的原结论

1. `_owned_streams` 以 `id(native_stream)` 为键，归属判定只能靠 `holder() is native_stream`；
2. 该表原先用**弱引用** ⇒ 对象被回收后 `id` 被新对象复用 ⇒ 误判为"本层拥有" ⇒ 越权销毁；
3. 证据：修复前 4/4 FAIL（`71/1/6`）→ 修复后 8/8 PASS（`72/0/6`）；
4. 处置：`base.py::_register_owned_stream` 由 `self._id_holder(...)`（弱引用）改为 `lambda: native_stream`（强持有）。

---

## 2. 第 1 步：证伪「弱引用 ⇒ 越权销毁」

判定代码是**严格同一性**（`base.py` L503–L512）：

```python
reg = self._reg("_owned_streams")
ent = reg.get(id(native_stream))
if ent is None:
    return False
handle, holder = ent
if holder() is not native_stream:      # ← 身份复核
    return False
```

| 场景 | 弱引用（原实现） | 强持有（"修法"） |
|---|---|---|
| 命中条目且对象**存活** | `holder() is native` → 销毁 | 同左 |
| 命中条目但对象**已回收** | `holder() is None` → **False，不销毁** | 该场景不会发生 |

⇒ 「弱引用 ⇒ 越权销毁」**在代码上不成立**。

### 2.1 本机判定实验（用真实 `base.py` 代码）

`../probes/l1_recheck_20261010/local_idreuse_test2.py` —— 以弱引用登记、回收对象、
**反复分配直到新对象拿到同一个 `id`**，再调**真实的** `release_stream()`：

```
登记: id(A)=0x10395fc50 holder 类型=ReferenceType
回收后 holder() = None
id 复用成功: id(B)=0x10395fc50 == id(A)
release_stream(B) = False            ← 身份复核正确拦下
场景二（同一对象，真拥有）: release_stream(C) = True   ← 判据有牙齿
```

> ⚠️ 首版误用 `object()` 做样本 —— 它**不可弱引用**（`weakref.ref` 抛
> `TypeError: cannot create weak reference to 'object' object`），`_id_holder` 会走
> `lambda: obj` 回退 ⇒ 样本取错。改用普通类实例后结论成立。

---

## 3. 第 2 步：交替对照（这一步**推翻了我自己的"修法无效"判断**）

`../probes/l1_recheck_20261010/switch_holder.py` 就地切换那一行，**A/B 交替**跑（消除时间漂移）。

### 3.1 审计路径（完整 78 项）

| 版本 | 次数 | 结果 |
|---|---|---|
| 强持有 | 3 + 15 | **18/18 全部 `OK 72 / FAIL 0 / SKIP 6`** |
| 弱引用（=「修复前」实现） | 3 + 15 | **18/18 全部 `OK 72 / FAIL 0 / SKIP 6`** |

⇒ **审计路径下两版都不复现 FAIL** ⇒ 原「修复前 4/4 FAIL → 修复后 8/8 全绿」属**时间混淆**，
不能作为修法有效的证据（这一点原判断成立）。

### 3.2 探针路径（忠实序列 + 只在命中登记表时记录，`probe_l1_truth.py`）

⭐ **交替跑 5 轮（弱/强相邻），结果完全分离**：

| 版本 | L1 判定 | `[hits]`（命中的登记条目） |
|---|---|---|
| **弱引用 ×5** | **5/5 `[FAIL]`** | 每次都命中 **1 条**：`{type:"Stream", holder_kind:"ReferenceType", held_type:"Stream", held_is_native:true}` |
| **强持有 ×5** | **5/5 `[OK]`** | **`[]`（从未命中任何条目）** |

原始输出（`../probes/l1_recheck_20261010/out/ab_weak_1.log` 与 `ab_strong_1.log`）：

```
weak   [hits] [{"id": 140694913984320, "type": "Stream", "handle": 94472371533040,
                "holder_kind": "ReferenceType", "held_type": "Stream", "held_is_native": true}]
       [reg] [[140694914362944, "ReferenceType", null]]
strong [hits] []
```

⇒ **两条结论**：
1. ✅ **holder 强度确实有因果作用**（在该序列下）：`5/5 FAIL` vs `5/5 OK`，且是**交替**跑 ⇒ 不是时间漂移。
2. ⚠️ **但原「根因」仍不成立**：hits 显示命中的条目里 `holder()` **有效**（返回对象而非 `None`），
   且那个对象**就是 native 本身**（`held_is_native=true`）—— 这不是"同一 id 被复用"能解释的形态。

---

## 4. 🔴 第 3 步：疑点已收敛到「默认流被登记」

### 4.1 现场事实

hits 里三个字段放在一起看：

| 字段 | 值 | 含义 |
|---|---|---|
| `type` | `Stream` | **L1 传给 `release_stream` 的对象**是默认路径的流（`torch.cuda.Stream()`） |
| `held_type` | `Stream` | 登记条目里存的对象**也是 `Stream`** |
| `held_is_native` | `true` | 两者是**同一个对象**（`is`） |
| `holder_kind` | `ReferenceType` | 该条目用的是**弱引用**持有器，且**未失效**（对象存活） |

⇒ 也就是说：**`_owned_streams` 里存在一条以「默认路径的流」为对象的登记条目。**

### 4.2 与代码的矛盾

全仓 `_owned_streams` 的**唯一写入点**是 `_register_owned_stream()`（`base.py` L485），
其**唯一调用点**是三家后端的 `_create_stream_raw(priority=<int>)`；
PPU 那一支（`backend.py` L482–L487）登记的 native 一律是 `torch.cuda.ExternalStream(...)`。

⇒ **`Stream` 类型的对象不该出现在这张表里。**

### 4.3 登记来源探针的结果（并暴露插桩扰动）

`../probes/l1_recheck_20261010/probe_reg_trace.py`（对 `_register_owned_stream` 打点）输出：

```
[L1] FAIL :: release_stream(默认路径的流) 返回 True
[REG ] register 调用（caller, 类型, id）:
        ('_create_stream_raw', 'ExternalStream', 140322179458320)
        ('_create_stream_raw', 'ExternalStream', 140322179820560)
[NOTE] note_stream_created 调用数 = 19  类型集合 = ['ExternalStream', 'Stream']
[表 ] _owned_streams 现存条目：
       key=140322179820560 handle=94358942332192 holder=ReferenceType held_type=None
```

⇒ 加了这层插桩后，**登记记录里只有 `ExternalStream`**（2 次），但 **L1 仍 FAIL** ⇒
说明插桩**改变了触发形态**（与 §3.2 的现场不一致）⇒ **该缺陷对扰动敏感，必须用"不插桩的现场"取据**。

### 4.4 因此，下一步该查什么（不是"持有器强度"）

1. **默认流的登记路径**：为什么 `torch.cuda.Stream()` 造出的对象会出现在 `_owned_streams`？
   —— 建议在**临时分支**上给登记条目加"来源标记"（登记序号 + `type(native).__name__` + 调用者），
   再跑探针序列；**该改动只用于诊断，不进产品代码**。
2. **为什么审计序列不触发**：探针与审计的差异点（探针在 `bk.release_stream` 上做了一次实例属性替换）
   ⇒ 需要确认"实例属性替换"本身是否改变行为（例如它使 `release_stream` 走 MRO 的方式不同）。
3. `K6` 的"建两条优先级流且从不释放"是否与触发相关（它在两个路径下都存在，但只有探针路径报 FAIL）。

---

## 5. 处置结论（对 `a6fade3` 的评价已修正）

1. ❌ **原「根因＝持有器强度」的解释不成立** —— 需从代码注释、提交说明与 4 份文档中撤下「根因已定位」的表述。
2. ✅ **但该改动本身不建议回退** —— 交替对照证明：**没有它，探针序列 5/5 FAIL**；有了它 5/5 OK。
   回退等于把一个**可复现的 FAIL** 放回代码。
3. ⚠️ **对外/对内表述改为**：「**L1 现象在可复现序列下已消除；根因（默认流为何出现在登记表中）尚未定位**」。
4. 🔴 **`L1` 仍列为未收口项**：审计路径 18/18 全绿，但**探针路径稳定 FAIL ⇒ 该契约仍有可违反路径**。
   下一步按 §4.4 取证。
5. **决策记录**：上一版本报告曾建议"回退该改动"，依据是"弱引用 3/3 全绿"。
   该依据被**交替对照 5/5 vs 0/5** 推翻 ⇒ **建议已撤回**（保留改动）。

---

## 6. 方法论教训（本轮最值钱的部分）

1. ⭐⭐ **跨条件对照必须交替（A/B/A/B），不能"先连跑 A 的 N 次、再连跑 B 的 M 次"** ——
   后者无法区分"变量效应"与"时间/环境漂移"。本轮正是踩了这个：分组跑导致把漂移当变量效应，
   写下「已修复」的错结论；而**交替跑**才让真实的因果关系显形（5/5 vs 0/5）。
2. ⭐⭐ **观测工具必须与真实执行路径一致**：上一轮 3 个 spy 探针都用
   `if sid in mod.INVASIVE_SIDS: continue`（**跳过**侵入项），而真实审计对侵入项走
   `_run_in_child()`（**子进程隔离**）⇒ 序列被改变 ⇒ 观测不可信（这 3 个探针已加警示注释）。
3. ⭐ **对扰动敏感的缺陷，要区分"插桩能看到的"与"不插桩才成立的"现场**：
   本缺陷在加登记插桩后登记记录变了（§4.3），说明必须两条腿走路 —— 先在不插桩下抓现场（§3.2），
   再用插桩验证假设，并明确标注哪次观测带扰动。
4. ⭐ **证据 stage 时不要"瘦身"**：上一轮把 pre/post 日志裁成每份 2 行，**丢掉了 detail**
   （`release_stream(...) 返回 True` 这类关键信息），本轮已补回完整 log/JSON（`l1_fix_20261010_out/full/`）。
5. ⭐ **同时间点的"本地改动"与"远端副本"必须显式核对**：`proto_train_leg.py` 的 `_DIST_BT_DEFAULT`
   修复（补 `ppu` 项）至今**未同步到容器**（远端 ctime=10:51，本地 mtime 亦 10:51）⇒
   今日训练腿是**显式传 `DC_DIST_BT=nccl`** 跑的（结论不受影响），但该修复需重新同步后验证。

---

## 7. 遗留与边界

- 未做跨后端复跑（P800 容器 stopped；910C / MLU590 不可达）。**保留该改动 ⇒ 仍应补跑四家 78 项。**
- 「探针序列」与「审计序列」的差异点尚未查明（§4.4 第 2 条）。
- 本轮全部实验都在同一容器、同一张卡（`CUDA_VISIBLE_DEVICES=0`）上完成；
  结论的**适用范围**仅限该实例，不得外推。
