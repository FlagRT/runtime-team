# 流优先级「回读」口径修复 + L5 判据入口补齐（2026-10-08 第十二轮）

> 作者：Kistich（hliu553）｜触发：核对「§1.10 的 L5 三家为何都行使不了」时**连查出三件事**
> 影响面：`cambricon` 后端实现（口径变更）· `base.py`（所有权登记表）· 职责审计 L5 判据 ·
> 离线自检（假驱动与注入面）· P800 与 MLU590 真机复跑
> ⚠️ 本轮**同时更正了 2026-10-08 第十一轮 m1 报告里的一处错误结论**（见 §5）。

---

## 0 三句话结论

1. **L5 本身就是契约 §1.10 规则 4**（「使用已销毁对象必须明确」）。它**不是能力缺口** ——
   P800 早在 **2026-09-30** 就已经真机行使过它（`R3c/R3d`），只是那一轮走的是**后端私有原语**路径，
   而职责审计的 L5 只走**统一面**⇒ 显示 SKIP ⇒ 读者被误导成「无覆盖」。**已修判据入口。**
2. **MLU590 的「能设置流优先级」此前是空转判据撑起来的假象**：本层读的是
   `torch.mlu.Stream.priority`，而它**只是构造参数的回显**（请求 −1 ⇒ 属性 −1，**设备侧其实是 3**）
   ⇒ 契约 §1.9 第 ⑥ 条「回读≠请求 ⇒ `RuntimeError`」在 MLU590 **永远不可能 FAIL**。
   **已把整条路径改为厂商 C API 口径**（与 kunlun 同构）。
3. 顺带修掉一个**长期登记但未修**的真缺陷：所有权登记表以 `id()` 为键 ⇒
   **id 复用会让 `release_stream()` 越权销毁厂商的流**。它此前"没发作"只是因为
   MLU590 从不产生「本层拥有」的流；口径一改就当场发作。

---

## 1 L5 断言的**真正内容**（用户问题的直接回答）

契约 **§1.10 流所有权与释放** 共五条规则，L5 对应**规则 4**：

| 规则 | 内容 | 谁在守 |
|---|---|---|
| 1 所有权判据 | 走「**厂商 C API 建流 + 包装成 torch 流**」的流**由本层拥有**；厂商/torch 原生路径（`torch.npu/cuda/mlu.Stream()`）造的流由**厂商拥有** | `owns_stream()` · L4 |
| 2 不得越权销毁 | 厂商拥有的流 ⇒ `release_stream` **必须 no-op 返回 `False`** | L1 / L3 |
| 3 泄漏纪律 | 本层拥有的流用完**必须显式释放**（设备流总数有限：910C 实测上限 1979） | L4 |
| **4 使用已销毁对象必须明确** | **已释放的流再使用**（`synchronize()` / `context()` / `record_stream()`）⇒ **`RuntimeError`**（文案含「已释放」），**不得静默失败** | **L5**（本层 `check_stream_usable()` 在使用点拦截） |
| 5 幂等 | 重复 `release_stream` 返回 `False`、不报错 | L2 |

**L5 断言的是「本层自有拦截逻辑」**（`backend.check_stream_usable()` 在 `Stream.synchronize/context/record_stream`
三个使用点上先跑）—— 它**与厂商无关**，但**要行使它，必须先有一条「本层拥有」的流**。
⇒ 所以 L5 的"能不能行使"完全取决于**该实例是否会产生本层拥有的流**，而不是取决于设备能力。

---

## 2 为什么三家原先都"行使不了"（逐家如实）

| 实例 | 统一面 `create_stream(priority=…)` | 是否产生「本层拥有」的流 | 原 L5 |
|---|---|---|---|
| **910C** | 不声明 `stream_priority_control` ⇒ 基类第 ④ 条抛 `NotImplementedError`；`_create_stream_raw(priority)` **也直接抛**（无 `ExternalStream`、`Stream(stream_ptr=…)` 静默忽略） | ❌ **确实不可能**（原语面也产不出） | SKIP（**如实**） |
| **P800** | 区间**退化单点** `(0,0)` ⇒ 基类第 ③ 条「单点等价放行」⇒ 返回 **torch 流**（厂商拥有） | 统一面 ❌ / **后端原语 ✅**（`cuStreamCreateWithPriority` + `ExternalStream` + 登记） | 原 SKIP → **本轮改为真行使 ✅** |
| **MLU590** | 区间非单点 ⇒ 走 `_create_stream_raw(priority)`，而它原先用 `torch.mlu.Stream(priority=…)` | 统一面 ❌（torch 拥有） | 原 SKIP → **本轮改为真行使 ✅** |

⭐ **关键**：P800 的「本层拥有」路径**一直存在且工作**（`r8_prio_release_kunlun.json` 的
R3c `release=True`、R3d `RuntimeError: 该流已被 release_stream() 释放（厂商句柄 0x5ac9f7230730）`），
只是**职责审计没走它**。⇒ 缺陷定性为「**判据入口不齐**」，不是「能力缺口」。

---

## 3 ⭐ 本轮修的四件事

### 3.1 MLU590：优先级路径统一到**厂商 C API 口径**（真缺陷）

**缺陷**：本层 `_stream_priority_read_raw` 读 `torch.mlu.Stream.priority`。三列实测对照：

| 请求 | `.priority`（**旧口径**） | `cnrtQueueGetPriority`（**设备真读数**） |
|---|---|---|
| 无参 / 0 | 0 | **4** |
| −1 | −1 | **3** |
| −3 | −3 | **1** |
| 7（越界） | 0 | 4 |

⇒ `.priority` 与请求**恒等**（在合法域内）⇒ 「回读 == 请求」是**同义反复**，
第 ⑥ 条**不可能 FAIL**。**这正是 910C「参数被静默丢弃」当初能被抓出来所依赖的那条判据，
而 MLU590 用的恰是同族的 torch 属性** ⇒ 判据空转。

**改法**（与 kunlun 同构，一处范式两家复用）：`range` / `create` / `readback` / `destroy`
**四处统一**走 CNRT：

| 面 | 旧 | 新 |
|---|---|---|
| `stream_priority_range()` | `torch.mlu.Stream.priority_range()` → `(0,-3)` | `cnrtDeviceGetQueuePriorityRange` → `(7,0)`（合法域 `0..7`，数值越小优先级越高） |
| `_create_stream_raw(priority)` | `torch.mlu.Stream(priority=…)` | `cnrtQueueCreateWithPriority` + **`torch.mlu.ExternalStream` 包装** + 登记所有权 |
| `_stream_priority_read_raw` | 读 `Stream.priority` 属性 | `cnrtQueueGetPriority`（**设备读数**） |
| `_destroy_stream_raw` | （无） | `cnrtQueueDestroy` |

**真机验收**（`probes/probe_stream_capi_cambricon.py`，`CAMB_STREAM_CAPI_PASS` 9/9）：

| 步 | 结果 |
|---|---|
| S1 头文件入口表 | ✅ `cnrt.h` 有 `cnrtQueueCreate/CreateWithPriority/Destroy/GetPriority/DeviceGetQueuePriorityRange` |
| S2 库导出符号 | ✅ `libcnrt.so.7.4.0` 有 **17 个**同族符号 |
| S3 真调一次 | ✅ 8 个档位（0..7）全部 `create rc=0` / **回读==请求** / `destroy rc=0` |
| **S4 能否包回 torch** | ✅ **`torch.mlu.ExternalStream` 存在且可用** |
| S5 包装后的流 | ✅ 真能承载算子（`sum=32.0`）；未登记的流 `owns=False`、release no-op |
| S6 生命周期 | 建 200 条 `torch.mlu.Stream(priority=…)` ⇒ **不同句柄仅 32 个 ⇒ torch 侧池化复用、不泄漏** |
| S7 丢弃包装对象后 | 底层队列仍可用（`cnrtQueueGetPriority` rc=0）—— 与"池化"一致 |
| **S8 三列对照** | ✅ **8 个档位「本层回读 == 设备 C API」全部一致**（torch 属性列**允许不等**，它就是被替换掉的旧口径） |

⭐⭐ **同一处修复还让一条原本"报喜"的判据开始"报忧"**：`D5 厂商路径的参数保留性` 现在报
**「直接 mlu 建流 priority=7 ⇒ 回读=4 ⇒ 厂商静默丢弃」** —— 即 **torch 侧路径确实把参数丢了**，
旧口径只是看不见。⇒ 修复前 MLU590 的「能设置」是**假象**；修复后是真事。

### 3.2 更正两处**未经验证的推测**（cambricon 后端注释）

| 原注释（长期） | 实测（2026-10-08） |
|---|---|
| 「本家**没有**独立的句柄式 C API（与 ascend / kunlun 不同）」 | ❌ **错**：`cnrt.h` + `libcnrt.so` 有完整 queue（寒武纪把 stream 叫 queue）原语；`ExternalStream` 也有 |
| （隐含）「无 `ExternalStream` 等价入口」（从 ascend 外推） | ❌ **错**：`torch.mlu.ExternalStream` 可用（S4/S5） |

⭐ 教训：**「另两家做不到 ⇒ 这家也做不到」的外推必须走三步判定**（头文件入口表 / 库导出符号 / 真调一次），
否则**注释会代替事实长期存在**，并误导下一轮的所有判断（本轮 L5 的误判就源于此）。

### 3.3 `base.py`：**id 复用**不得让 `release_stream` 越权销毁（长期登记项，本轮发作）

**缺陷**：`_owned_streams` / `_released_streams` / `_stream_ctx` 三张表都以 `id(obj)` 为键，
而 **`id()` 在对象回收后会被复用** ⇒ 一条**全新**的流可能撞上"已登记/已释放"那条的 id。三类后果：
① `release_stream()` **越权销毁厂商拥有的流**（契约 §1.10 规则 2 明令禁止 —— "比泄漏更糟"）；
② 新流被误判"已释放"而**误拦**；③ 新流被误判"绑定在已销毁上下文上"而**误拦**。

**暴露条件**：只有当实例**真的会产生「本层拥有」的流**时 ① 才有机会发作 ⇒
MLU590 在 3.1 之前**永远碰不到**。3.1 一落地，MLU590 的 **L1 当场 FAIL**
（`release_stream(默认路径的流) 返回 True`）。此前它只作为"已知未修风险"登记在册。

**修法**：登记时一并存**身份持有器**（优先 `weakref`，类型不支持则退化为强引用 lambda），
查询/释放时**按对象身份复核**；id 复用命中时**既不得销毁也不得报 True**，
且原条目**保留在册**（它记录的是"有一条本层拥有的流被回收却没释放"这个**泄漏事实**，不掩盖）。

**非空转验证**（离线，无需设备）：直接注入该情形 —— 把 a 的条目搬到 b 的 id 下（等价于"b 拿到 a 的 id"）：

```
[PASS] id 复用：owns_stream 必须按身份判定（不得把一条新流当成已登记的那条）  owns_stream(_b)=False
[PASS] id 复用：release_stream 不得越权销毁（必须如实返回 False）              release_stream(_b)=False
```

### 3.4 L5 判据入口补齐（并踩到一个"验证方式"的坑）

- **补路径 B**：统一面拿不到拥有流时，回落到**后端私有原语**（与探针 R3 同法），
  并**逐条如实回报**走的是哪条、另一条为何不通。⇒ **P800 的 L5 由 SKIP 变为真行使并通过**。
- ⚠️ **踩的坑**：路径 B 的对象是**厂商原生流**，release 之后**已被销毁** ——
  我原先照样调 `st.synchronize()`，那是**在已释放对象上调用原生 API**（未定义行为），
  **P800 实测直接把整个进程打断：`rc=1` 且无任何输出**（看上去像"审计莫名消失"）。
  ⇒ 改为**按路径选择验证方式**：路径 A 用包装的 `synchronize()`（内部先走使用点拦截），
  路径 B 用**本层** `check_stream_usable()`（探针 R3d 本来就是这么写的 —— 本例是没照抄现成范式而踩的坑）。

---

## 4 真机复跑结果（两台）

| 项 | MLU590（`cambricon`） | P800（`kunlun`） |
|---|---|---|
| 离线自检 | **97 / 0 / 0** | **108 / 0 / 1** |
| 冒烟 | 46 / 0 | 46 / 0 |
| conformance 13 + 推理 6 | `CONFORMANCE_PASS` ×2 | （同款，未变） |
| 契约不变式 I1–I4 | `CONTRACT_INVARIANTS_PASS` | — |
| **职责审计（78 项）** | **`DUTY_RESPONSE_PASS` 62 / 0 / 16** | **`DUTY_RESPONSE_PASS` 68 / 0 / 10** |
| **非空转验证** | **`SELFCHECK_DUTY_EXT_PASS` 26 抓到 / 13 不适用 / 0 未抓到** | **32 / 7 / 0** |
| **释放/所有权探针** | **`STREAM_RELEASE_CONTROL_PASS` 9 / 0 / 1**（**R3c/R3d 首次真行使**） | **10 / 0 / 0**（R3c/R3d 持续通过） |
| 优先级 API 探针 | `STREAM_PRIORITY_API_PASS` 7/7 | 6/6 |
| 全 78 项 `--only` rc 检查 | 无静默退出 | 无静默退出 |

**L5 现在的实测文案**（各自走了不同路径，都**如实回报**）：

```
MLU590: [OK] L5 经**A 统一面 create_stream(priority=…)**行使，验证方式 = 包装的 synchronize()（内部先走使用点拦截）；
             RuntimeError: 该流已被 `release_stream()` 释放（厂商句柄 0x55b562baef40）⇒ 不可继续使用
P800  : [OK] L5 经**B 后端原语 _create_stream_raw(…)【取证用途…】**行使，
             验证方式 = 本层使用点拦截 check_stream_usable()（原生流已销毁，不得调其原生方法）；
             RuntimeError: 该流已被 `release_stream()` 释放（厂商句柄 0x5f5661405be0）⇒ 不可继续使用
```

**离线`[8b]`段的配套改动**（否则新链路在离线是空转）：
① 假驱动**同时挂 `cu*` 与 `cnrt*` 两族符号名**（同一份实现；厂商符号名只是"被取用的名字"）；
② **注入面也要覆盖两族** —— 只换一族会让另一族的后端**压根调不到被注入的函数** ⇒
判据空转并把结果读成「未传播」（**假失败**，本轮实测踩到）。离线自检：ascend **90/0/1** ·
kunlun **108/0/1** · cambricon **97/0/0** · 对称性 **7/0**。

---

## 5 ⚠️ 更正第十一轮（m1）报告里的一处错误结论

m1 报告 §4.2 / 台账 **E2** / STATUS / 看板 / skill 里写过：

> ❌ 「**910C / P800 / MLU590 三家现役实例都行使不了 L5** ⇒ 契约 §1.10 该条款目前只有离线覆盖」

**这句是错的**（两处错）：

| 原表述 | 事实 |
|---|---|
| 「P800 行使不了 L5」 | ❌ **错**。P800 在 **2026-09-30 r8** 就通过了 **R3c/R3d**（`P800/probes/r8_regress_kunlun_20260930_out/r8_prio_release_kunlun.json`）—— 走**后端原语**路径。只是**职责审计的 L5 没走那条路**。 |
| 「该条款只有离线覆盖」 | ❌ **错**。至少 P800 有真机覆盖；本轮起 MLU590 也有；**只有 910C 如实没有**（其 `_create_stream_raw(priority)` 直接抛 `NotImplementedError`，且无 `destroy` 原语 ⇒ 原语面也产不出拥有流）。 |

**正确的表述**：**缺陷是「判据入口不齐」**（同一契约条款有两个探针，一个走统一面 SKIP、一个走原语 PASS
⇒ 读者被误导），**已修**（L5 补路径 B 并如实回报走法）。

**台账 E2 同步降级/更正**：由「三家都行使不了」改为「**判据入口不齐**（已修）」；
910C 的"如实不适用"保留为**覆盖缺口登记**（它是真的没有该路径，且已在 `_create_stream_raw`
里显式抛错 + 上游诉求 `ExternalStream`）。

**台账 D1 的处置**：m1 报告称 D1「已收尾」——**该结论建立在 3.1 的空转回读之上**，
按纪律**降级为"经修复后重新收尾"**：口径修复后的 MLU590 复跑
（优先级 API 7/7、区间 `(7,0)`、请求 0/7 回读一致、K6 端点通过、释放探针 R3c/R3d 通过）
才是 D1 的有效依据。

---

## 6 边界（不得外推）

1. 全部结论在**本栈**取得：MLU590 = 驱动 6.2.29 / `neuware4.4.3`（torch 2.7.1 / torch_mlu 1.29.2）；
   P800 = XPytorch + XRE（torch 2.9.0+cu129）。
2. **「能设置」≠「有效果」**：本轮仍**未**做调度效果对照（台账 D2 未做）。
3. **档位口径变更**已如实记录：MLU590 声明的区间由 torch 口径 `(0,-3)` 改为**设备口径** `(7,0)`；
   两套口径**不是恒等映射**（torch 请求 0 ⇒ 设备 4）。引用旧证据（m1 轮）时必须连同口径一起引。
4. **行为变更**：MLU590 的 `create_stream(priority=…)` 现在返回**由本层拥有**的流 ⇒
   调用方须按契约 §1.10 规则 3 显式 `release_stream()`（否则泄漏一条设备级队列）。
5. `S7`（丢弃包装对象后底层队列仍可用）的解读是**"池化复用"**而非"泄漏"：`S6` 实测 200 次建流
   只产生 32 个不同句柄 ⇒ torch 侧有流池。

---

## 7 证据清单

| 内容 | 路径 |
|---|---|
| MLU590 流原语三步判定 + 可包装性 + 生命周期 + **三列对照**（9 项 JSON） | `MLU590/probes/r12_capi_20261008_out/m1_capi_cambricon.json` |
| MLU590 职责审计 / 非空转 / 优先级 API / 释放探针 | 同目录 `m1_duty_cambricon.json` · `m1_selfcheck_duty_ext.json` · `m1_prio_api_cambricon.json` · `m1_prio_release_cambricon.json` |
| P800 同轮三项 | `P800/probes/r12_20261008_out/r12_duty_kunlun.json` · `r12_selfcheck.json` · `r12_prio_release_kunlun.json` |
| 新增探针 | `prototype/probes/probe_stream_capi_cambricon.py` |
| 改动面 | `prototype/runtime/backends/cambricon/backend.py` · `runtime/backends/base.py` · `scripts/duty_response_audit.py` · `scripts/backend_offline_check.py` · `probes/selfcheck_duty_audit_ext.py` |
