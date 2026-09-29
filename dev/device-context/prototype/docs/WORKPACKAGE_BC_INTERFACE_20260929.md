# 工作包 B / C · 接口落地与真机验证（2026-09-29）

> 定位：**设计 + 实现 + 判据 + 实测记录**。对应《薄弱环节补做计划》
> （`DEVICE_CONTEXT_GAP_CLOSURE_PLAN_20260929.md`）的工作包 **B（内存域句柄与生命周期）**
> 与 **C（设备上下文生命周期）**，本轮**先落 910C 与 P800 两家**。
> 边界：结论只在**各自档位/环境**成立（910C：cann9.0.0 / torch_npu 2.11.0；
> P800：XPU-RT 5.0.21 / xpy torch 2.9.0 / 卡 4）。**MLU590 本轮未探测**（见 §6）。
> 证据：`../../910C/probes/probe_bc_contract_ascend_20260929.json`、
> `../../P800/probes/probe_bc_contract_kunlun_20260929.json`

---

## 0 一句话结论

**接口、实现、判据、真机验证四件都齐了**：
910C 上 **B1/B3/B4/C1/C2/C3 六项全通过**，P800 上 **B1/B3/B4 三项全通过**（C 三项**如实不具备**，非失败）；
离线判据 5 处**注入缺陷**均能被抓到（非空转）；**未闭合项已如实登记**（§6）。

**过程中发现 3 处"厂商原语与直觉不符"的硬事实**（§3），其中 1 处是
**`hasattr` 为真但一调就报错**——正是本方向台账第 6/11 条那个家族的又一例。

---

## 1 交付一览（能力 × 判据 × 两实例结果）

| # | 能力键 | 对外 API | 判据（先写死） | 910C | P800 |
|---|---|---|---|---|---|
| **B-1** | `memory_alloc` | `runtime.allocate(size_bytes)` / `runtime.free(handle)` | 申请有真实占用；`allocate/free` 成对且**无泄漏**；**二次释放 / 未见过句柄 ⇒ 必须报错**；`size<=0` / 非整数 ⇒ `ValueError` | ✅ | ✅ |
| **B-2** | `memory_alloc_stat` | `memory_stats()` 第 4 键 `allocated_mb` | 三键语义**不变**、`allocated_mb` 只增；声明了就**必须给得出**；不得出现未登记键 | ✅ | ✅ |
| **B-3** | `record_stream` | `Stream.record_stream(tensor)` | 声明 ⇒ 走**原生路径**（零退化）；未声明/张量不支持 ⇒ 走**保守同步**且**不抛错**、可观测（`degradations`） | ✅ | ✅ |
| **B-4** | —（常开审计） | `info()["native_accesses"]` / `info()["degradations"]` | 公开读 `.native` ⇒ 计数 +1；**层内部路径不得计数** | ✅ | ✅ |
| **C-1** | `context_lifecycle` | `context_create/set/destroy/count` | 创建 ⇒ 计数 +1；销毁 ⇒ 回落；**重复销毁 / 非本层句柄 ⇒ 必须报错** | ✅ | ➖ 如实不具备 |
| **C-2** | 同上 | 使用点拦截 | **销毁上下文后使用其流 ⇒ 必须如实报错**（不得静默成功） | ✅ | ➖ |
| **C-3** | 同上 | `recover_device()` 增键 | 契约五键**不变**，增 `context_supported/context_count/context_recreated`；多上下文**互不串** | ✅ | ➖ |

> `➖` = 后端**未声明**该能力 ⇒ 调用**如实报错**（`NotImplementedError`），探针如实 SKIP。
> 这是"如实不具备"，**不是**"已确认该芯片不具备"（见 §6）。

---

## 2 接口设计要点（为什么这么定）

### 2.1 内存句柄（B-1）

- **只做句柄语义**：申请/释放两件事，**不做池化 / 碎片 / 峰值 / 扩容**（那些属显存方向）。
- ⭐ **句柄里不放厂商指针**：公共字段固定为
  `{handle_id, kind, backend, ordinal, size_bytes}`（`MEMORY_HANDLE_KEYS`）。
  暴露裸指针等于绕开 `.native` 逃生舱纪律，会把"换芯片不改代码"的主张**悄悄**破坏掉。
- ⭐ **把厂商的"静默"变成显式错误**：实测 pyACL `acl.rt.free()` 对**二次释放静默返回 0**；
  昆仑芯则在 torch 层报错。两家的**原始行为相反**，而契约要求一致 ⇒
  **统一层自己登记句柄、自己拦截**，于是"重复释放必须报错"这条判据**不再依赖任何厂商**。
  （离线判据特意把 stub 的 `free` 也做成静默 ⇒ 谁把登记表拿掉，判据就 FAIL。）

### 2.2 `memory_stats` 第四键（B-2）

- 规范三键 `total_mb / used_mb / free_mb` **语义与口径不变**（以本方向为准，供显存方向采集）；
- 新增 `allocated_mb` = **分配器视角**（本进程已分配量）；**取不到就省略该键**，不填 0 冒充；
- 判据由"**精确等于三键**"改为"**三键齐备 + 只增键 + 未登记键 FAIL**" ——
  这不是放宽，是把"合法扩展"与"键名漂移"分开判。

### 2.3 `record_stream` 能力位与保守路径（B-3）

- 上层可**提前预判**（`supports("record_stream")`），而不是等运行时 `getattr` 失败；
- 后端未声明、或张量本身不支持时 ⇒ **保守同步路径**（同步当前设备后放行）+
  `warnings.warn` + 计入 `degradations`。
  **刻意不再抛 `AttributeError`**：跨流内存被提前回收是"偶发数据错乱"，比降级慢一点危险得多。

### 2.4 `.native` 逃生舱审计（B-4）

- 修订建议 §3 要求"取用即视为绑定该厂商、且要能事后溯源" ⇒
  **公开属性 `.native` 计数**，**层内部改走私有 `_native_obj`（不计数）**。
  两类计数的语义相反，因此**分开**：`.native` = 破坏可移植性的坏账；
  `degradations` = 用性能换正确性的可接受降级。

### 2.5 设备上下文生命周期（C）

- 修正一处**长期混淆**：原型的 `stream_context` 是**切流**的上下文管理器，
  与"设备上下文的创建/销毁"是**两件事**；而 `recover_device(mode="real")` 内部其实会
  `destroyContext → ResetDevice → 重建`——**执行了但完全不暴露**。
- **安全契约**：`context_destroy` **只接受本层创建的句柄**。
  非本层句柄（尤其是**进程默认上下文**）**一律拒绝并报错** —— 误毁默认上下文会让整个进程的设备不可用。
- **绑定语义（C-2）**：实测 910C 上"销毁上下文后使用其流"**当场不报错**，
  直到**进程退出清理阶段**才暴露 `stream not in current ctx`（107003）⇒
  本层在 `create_stream()` 时登记"该流属于哪个本层上下文"，并在
  `Stream.context()` / `Stream.synchronize()` 两个**使用点**主动拦截（见 §5.3 的对照取证）。
- `recover_device` 的上下文维度在**基类统一附加**（写成三遍必然三套口径），**只增不改**。

---

## 3 三处"厂商原语与直觉不符"的硬事实（本轮新增认知）

| # | 直觉写法 | 实测真相 | 后果 |
|---|---|---|---|
| 1 | `torch.npu.caching_allocator_alloc`（`hasattr` 为 **True**） | 该名字在本栈上只是**继承了 `torch.cuda` 的实现**（torch_npu 未覆写）⇒ 调用走 `torch.cuda.current_device()` → `_cuda_init()`，直接报 **`Found no NVIDIA driver on your system`** ⇒ **`hasattr` 为真 ≠ 可用** | 910C 上必须改用 **pyACL `acl.rt.malloc/free`**；本判据家族再次命中（台账第 6/11 条） |
| 2 | `caching_allocator_free` | 昆仑芯/910C 上**都不存在**；正确名是 **`caching_allocator_delete`** | 若照抄 CUDA 文档写 `_free`，会静默找不到原语 |
| 3 | `acl.rt.create_context()` 返回句柄 | 实际返回 **`(handle, ret)` 元组**（`set_context`/`destroy_context` 则直接吃 handle） | 直接 `set_context(create_context(...))` 会 `args parse failed` |

> 另有一条**接口内部**的同类问题，被本轮**新判据当场抓到**：上下文句柄初版写成 `context_id`，
> 而取句柄的公共口只认 `handle_id` ⇒ **上下文永远销毁不掉**。
> 已统一为 `handle_id` + `kind`（`memory`/`context`），并加"跨种类误用必须报错"判据。
> —— 同一概念两个对外形态，正是台账 ⑫b 家族的坑。

---

## 4 判据与非空转验证

**离线自检（无设备）新增 [9] 段**：内存句柄与设备上下文，覆盖
"声明即承诺 / 成对释放 / 负向必须报错 / 保守路径 / 审计隔离 / 只增不改"六类。

判据数变化：**ascend 40 → 64**（+1 跳过）· **kunlun 45 → 65**（+1 跳过）· **cambricon 45 → 55**。

**非空转验证（注入缺陷 ⇒ 判据必须能 FAIL）**

| 注入 | 预期命中判据 | 实测 |
|---|---|---|
| **D1** 去掉句柄登记表拦截（模拟"厂商静默被透传"） | 二次释放 / 未见过句柄 / 重复销毁 / 非本层句柄 | **4 条 FAIL** ✅ |
| **D2** 上下文句柄字段名漂移（`context_id`） | context_create 公共字段 + 使用点 | **2 条 FAIL** ✅ |
| **D3** 去掉保守同步路径（改为直接抛错） | record_stream 保守路径 | **1 条 FAIL** ✅ |
| **D4** 去掉 `.native` 审计计数 | `.native` 取用计数 | **1 条 FAIL** ✅ |
| **D5** 去掉上下文↔流的使用点拦截 | 销毁后使用其流 | **1 条 FAIL** ✅ |

> 附带收紧：负向判据原为"只要报错就算过"，会把 `KeyError` 这类**崩溃**误判为通过 ⇒
> 已改为**必须是契约级 `ValueError`**（D1 因此由"崩溃"变成"4 条明确 FAIL"）。

---

## 5 真机验证（910C + P800）

### 5.1 关键数据

| 项 | 910C（ascend） | P800（kunlun） |
|---|---|---|
| 句柄公共字段 | `['backend','handle_id','kind','ordinal','size_bytes']`，**无 `ptr`** ✅ | 同 ✅ |
| `allocate(8 MiB)` 后设备空闲 | **62593.7 → 62583.7 MB（−10.0）** ✅ 真实占用 | **98272 → 98252 MB（−20.0）** ✅ |
| `memory_stats()["allocated_mb"]` | 0 → **0** → 0 ⚠️ **如实**：pyACL 分配**不经 torch 分配器** | 0 → **8** → 0 ✅（分配器视角真实反映） |
| `free` 后设备空闲 | **不回落**（62583.7 持平）⚠️ pyACL 内存池语义 | 持平（分配器已归零） |
| 二次释放 | `ValueError`（本层拦截）✅ | `ValueError`（本层拦截）✅ |
| `allocate(0)` | `ValueError` ✅ | `ValueError` ✅ |
| `record_stream` 真机 | **原生路径（无退化）** ✅ | **原生路径（无退化）** ✅ |
| `.native` 审计 | 公开读=1 / 内部路径=**0** ✅ | 同 ✅ |
| 上下文生命周期 | 计数 0→1→0；`compute_in_ctx`=4.0；销毁后计算仍可用 ✅ | ➖ 未声明 |
| **C-2 绑定语义** | 统一层：**`RuntimeError` 如实拦截** ✅<br>厂商原生对照：**未报错（静默成功）** | ➖ |
| **C-3 多上下文隔离** | 同时 2 个；A/B 各自取值 4.0；销毁 B 后回 A 仍 4.0；全销毁后设备可用 ✅ | ➖ |

### 5.2 汇总判定（探针自评）

| 判定 | 910C | P800 |
|---|---|---|
| `B1_allocate_free_pairs` | ✅ | ✅ |
| `B3_record_stream_native_path` | ✅ | ✅ |
| `B4_native_audit_isolated` | ✅ | ✅ |
| `C1_context_lifecycle` | ✅ | ➖ 未声明 |
| `C2_binding_intercepted` | ✅ | ➖ |
| `C3_multi_context_isolated` | ✅ | ➖ |
| **`PASS`** | **true（6/6）** | **true（3/3）** |

### 5.3 一处值得单独说的取证（C-2）

同一进程内、同一个被销毁的上下文上建的流：

- **统一 API**（`stream.context()`）⇒ **`RuntimeError` 如实拦截**（本层登记了"流属于哪个上下文"）；
- **厂商原生流**（直接 `torch.npu.Stream()`）⇒ **`未报错（静默成功）`**，直到进程退出清理阶段才暴露 107003。

⇒ 契约里"**销毁后使用其流的行为必须明确**"这条，**不靠厂商**是本层自己补上的 —— 这是"统一层"在真实故障面上的一次具体兑现。

---

## 6 未闭合项与边界（如实登记，不补零）

| 项 | 状态 | 说明 |
|---|---|---|
| **MLU590（cambricon）** | ⛔ **本轮未探测** | 用户指定先落 910C 与 P800。**4 个新能力键在 cambricon 上全部未声明** ⇒ `info()["supports"]` 如实呈现为 `False`，调用如实报错。**下一轮**按同一探针取数后决定是否声明。 |
| `allocated_mb` 在 **910C** 上不反映 `acl.rt.malloc` | 🟡 **如实口径** | pyACL 分配绕过 torch 缓存分配器 ⇒ 该字段在 ascend 上只反映 **torch 侧**分配量。`memory_alloc_stat` 的声明**仅承诺"可给出该字段"**，**不承诺**反映 pyACL 分配。占用判据在 ascend 上用 `mem_get_info`。 |
| pyACL `free` 后设备空闲**不回落** | 🟡 **厂商池语义** | 未当作我方缺陷；判据只要求"申请后占用可见"，**不要求**"释放后回落"。 |
| `recover_device` 的 `context_recreated` | 🟡 **由实现如实回报**（原为派生字段，**2026-09-29 已修**） | 语义 = "本次是否**真的执行了**销毁/重建上下文的路径"。⚠️ 原派生条件 `mode == "real" and recovered` **用错了上游事实**（`recovered` = 设备当前可用）⇒ 在「健康设备上调 real」（`detail` 自写"无需重建"）与「未声明 `recovery_real` 的两家」上都**误报 `True`**（台账第 21 条）。现由 `conformance/recovery.py::last_rebuild_path()` 给出事实、后端经 `_rebuilt` 回报，**未回报视为未重建**。 |
| C 的 `context_set` | 🟢 已实现 | 计划里未列，但"切回某个上下文"是隔离验证的前提 ⇒ 作为 C 的最小可用集一并实现。 |
| 多进程 / 多卡下的 `mode="real"` 压测 | ⛔ 未做 | 已知挂账（与工作包 C 风险表一致）。 |

---

## 7 一键复跑

```bash
# 910C
ssh 910C 'docker exec dc-lean-910c-20260929 bash -lc \
  "cd /mnt/raid/hliu553/dc_regress_20260929/prototype && \
   /mnt/raid/hliu553/venvs/venv-infer-a/bin/python probes/probe_bc_contract.py --backend ascend \
   --out /mnt/raid/hliu553/dc_regress_20260929/out_bc/probe_bc_ascend_20260929.json"'
# P800（注意：卡 1 为已知故障卡，勿用）
ssh P800 'docker exec hliu553-device-context-p800 bash -lc \
  "source /root/miniconda/etc/profile.d/conda.sh && conda activate python310_torch29_cuda && \
   cd /workspace/dc_regress_20260929/prototype && CUDA_VISIBLE_DEVICES=4 \
   python3 probes/probe_bc_contract.py --backend kunlun \
   --out /workspace/dc_regress_20260929/out_bc/probe_bc_kunlun_20260929.json"'
# 无设备判据（本地即可）
python3 prototype/scripts/backend_offline_check.py --backend ascend|kunlun|cambricon [--all]
```

---

## 8 改动清单

**接口层（芯片无关）**
- `runtime/backends/base.py`：4 个新能力键位 + `allocate/free/memory_handle_count` +
  `context_create/set/destroy/count/current_context_id` + `note_stream_created` / `check_stream_usable` +
  `peek_current_device` / `conservative_stream_sync` + `.native`/退化审计 + `recover_device` 统一包装（`_recover_device_impl`）+ 句柄登记表与 `kind` 校验。
- `runtime/api/stream.py`：`_native_obj`（内部不计数）/ 公开 `.native`（计数）/ 能力位感知的 `record_stream` 保守路径 / 使用点拦截。
- `runtime/__init__.py`：转发 `allocate/free/memory_handle_count/context_*/native_accesses/degradations`，导出 `MEMORY_HANDLE_KEYS` / `CONTEXT_HANDLE_KEYS`。

**后端**
- `ascend`：声明 4 个能力；`pyACL malloc/free` + `create/set/destroy_context` 实现；`allocated_mb`；`peek_current_device`；`_recover_device_impl` 改名。
- `kunlun`：声明 `memory_alloc`/`memory_alloc_stat`/`record_stream`；`caching_allocator_alloc/delete` 实现；`allocated_mb`；`peek_current_device`；改名。
- `cambricon`：**4 个新能力键全部未声明**（下一轮探测）；`peek_current_device`；改名。

**判据 / 探针**
- `scripts/backend_offline_check.py`：`memory_stats` 判据随契约更新（三键齐备 + 只增键 + 未登记键 FAIL）+
  新增 `[9]` 段（含刻意"静默"的 stub 原语）+ 6 条负向判据收紧为"必须是 `ValueError`"。
- `probes/probe_bc_contract.py`：**新增**真机契约探针（5 组、逐组独立子进程、自带 verdict 汇总）。

---

## 9 第 3 轮全套回归（共享层改动之后 · 910C + P800）

B/C 改的是**共享层**（`backends/base.py` / `api/stream.py` / `runtime/__init__.py`）
⇒ 按「**按破坏面覆盖**」纪律，两家实例各跑一遍**全套**（与 r2 同口径）。

| 判定项 | 910C（r3） | P800（r3） |
|---|---|---|
| 离线契约自检 | **64 / 0 / 1 跳过** | **65 / 0 / 1 跳过** |
| 跨后端对称性 `--all` | 5 / 0 | 5 / 0 |
| 组件冒烟 | **52 / 0** | **46 / 0** |
| conformance 13 + 推理 6 | 13/13 + 6/6 | 13/13 + 6/6 |
| 职责响应审计（39 sub-part） | **39 / 0 / 0** | **36 / 0 / 3** |
| 错误注入→恢复闭环 | 5 / 0 / 0 | 5 / 0 / 0 |
| 工作包 A 功能等价性 | 6 / 6 | 6 / 6 |
| 多流语义 / 配额 | 8/8 · 3/3 | 8/8 · 3/3 |
| 结论 | ✅ **无回归** | ✅ **无回归** |

**⭐ 这轮抓到的唯一回归（已修）**：910C 首跑 `smoke` **rc=1** —— 冒烟 `[4]` 的
`info 含 name/device_type/capabilities` 断言用的是**精确键集**，
而工作包 B 给 `info()` 增了 `native_accesses` / `degradations` 两个审计字段
⇒ **合法扩展被判成失败** ⇒ 判据改为「**至少含三键**」，并注明"这不是放宽：
精确键集会把每一次合法的契约扩展都判成失败"（与 `memory_stats` 结构那条**同一类**）。修后两台重跑全绿。

> 教训（与 ⑫c 同族）：**「精确键集」式判据在契约走"只增不改"路线时必然误报。**
> 写这类判据前先问一句：**这个对象的键集会只增吗？** 若会 ⇒ 用「**至少含** + **未登记键 FAIL**」的组合。

证据：`../910C/probes/*_ascend_20260929_r3.*`（17 份）· `../P800/probes/*_kunlun_20260929_r3.*`（17 份）。

---

## 11 ⭐ C 项在 P800 的补足：从「如实不具备」到「只读观测」（2026-09-29 第五轮）

### 11.1 起因与更正

§3 曾把 P800 未声明 `context_lifecycle` 的原因记为
「**XPytorch 兼容层未暴露上下文原语**，与 `recovery_real` 同因」。**该表述不准确，现更正**：

| | 原表述 | 实测（四组判别实验） |
|---|---|---|
| 驱动层有无上下文 API | ~~无~~ | **有完整的 `cuCtx*`（21 个）**，就在 XPytorch 实际加载的 `libcuda.so.1`（= `libxpucuda.so.515.58.kunlun`，415 个 `cu*` 符号）里 |
| 是否真能创建 | —— | **能**（`cuCtxCreate_v2` 返回真句柄；`push/pop current` 语义成立） |
| 谁在用上下文 | —— | **XPytorch 自己在用**（torch 初始化后 `cuCtxGetCurrent` **非 0**） |
| 那为什么不能做生命周期 | —— | **平台只允许一个上下文**（第二次 create `rc=2`）+ **由框架自建** + **本层抢先去建会破坏框架** |

最后一条是关键硬证据：在 torch 之前建上下文 ⇒ torch 报
`AcceleratorError: CUDA error: invalid device ordinal`；
**销毁本层建的上下文后 torch 立即恢复（`512.0`）**。

### 11.2 处置

- **新增能力键 `context_query`**（只读观测），与 `context_lifecycle` **分开声明**；P800 **声明**，
  910C / MLU590 **如实未声明**（返回 `queryable=False` + 具体原因）。
- **本层实现绝不调用 `cuCtxCreate_v2` / `cuCtxDestroy_v2`** —— 只调
  `cuCtxGetCurrent` / `cuCtxGetDevice` / `cuCtxGetFlags`（真机证明只读安全）。
- 库按 **soname `libcuda.so.1`** 惰性加载，**不用副本路径**
  （踩过：从 `triton/backends/xpu/xpu3/so/` 取同名库 ⇒ 版本错配 ⇒ torch `CUDA_ERROR_NOT_INITIALIZED`）。

### 11.3 验证

- 真机 C4 组：`{queryable: true, present: true, ordinal: 0, flags: 8, managed_by: "external", reason: ""}`；
  **`compute_before = compute_after = 512.0`** ⇒ **只读无副作用**；verdict `C4_context_query_readonly = true`。
- 离线自检 **+6 条**判据（4 条通用 + 2 条「有上下文」分支）；判据数
  **ascend 64→68 · kunlun 65→71 · cambricon 55→59**。
- ⭐ **非空转验证暴露一个真实缺口**：离线无设备 ⇒ 只走 `if not raw` 分支，
  「丢掉 `managed_by`」「声明了却给 `queryable=False`」两处注入**都抓不到**
  ⇒ 补**可控桩**构造"有上下文"情形后，两处均能 FAIL。**能造可控假原语就别 SKIP**。
- 共享层改动后的 r4 全套回归：**910C 10 项全绿 · P800 10 项全绿**（见 §12）。

### 11.4 专项报告

`../../P800/docs/KUNLUN_CONTEXT_SEMANTICS_20260929.md`（含四组实验原始数据、资料出处、5 条边界）。

## 12 第 4 轮全套回归（`context_query` 落地后 · 910C + P800）

| 判定项 | 910C（r4） | P800（r4） |
|---|---|---|
| 离线契约自检 | **68 / 0 / 1 跳过** | **71 / 0 / 1 跳过** |
| 跨后端对称性 `--all` | 5 / 0 | 5 / 0 |
| 组件冒烟 | **52 / 0** | **46 / 0** |
| conformance 13 + 推理 6 | 13/13 + 6/6 | 13/13 + 6/6 |
| 职责响应审计（39 sub-part） | **39 / 0 / 0** | **36 / 0 / 3** |
| 错误注入→恢复闭环 | 5 / 0 / 0 | 5 / 0 / 0 |
| 工作包 A 功能等价性 | 6 / 6 | 6 / 6 |
| 多流语义 / 配额 | 8/8 · 3/3 | 8/8 · 3/3 |
| 结论 | ✅ **无回归** | ✅ **无回归** |

证据：`../910C/probes/*_ascend_20260929_r4.*`（15 份）· `../P800/probes/*_kunlun_20260929_r4.*`（15 份）。
