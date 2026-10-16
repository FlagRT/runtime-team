# 职责响应审计 · MLU590（`cambricon`）—— 39 项 sub-part 逐项实测

> **日期**：2026-09-28 ｜ **维护**：device-context（子方向 1）
> **判据来源**：《运行时层接口约定 · 设备上下文章节》`../../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md`
> （§1.1 后端选择 / §1.2 设备 / §1.3 流与事件 / §1.4 错误翻译 / §1.5 状态恢复 / §2 三支撑方法 / §3 两条硬纪律）
> **工具**：`../../prototype/scripts/duty_response_audit.py` ｜ **三实例汇总报告**：`../../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md`
> **本实例原始证据**：`../probes/duty_audit_cambricon_20260928.json`

---

## 1. 结论

**`DUTY_RESPONSE_PASS`：OK 36 / FAIL 0 / SKIP 3（共 39 项）** —— **零 FAIL**。

| 判定 | 含义 | 本实例 |
|---|---|---|
| `OK` | 调用成功**且**返回值/副作用符合接口约定 | **36** |
| `FAIL` | 不响应，或响应但不符合契约（字段缺失、静默降级） | **0** |
| `SKIP` | 该能力**被本后端如实声明为不具备**，且调用被主动拦截 | **3**（不是缺口） |

---

## 1.1 口径更新（2026-10-08）：**本文件是旧口径，需按 78 项重跑**

> ⚠️ 本文件 §1 的「**36 / 0 / 3（共 39 项）**」是 **2026-09-28 的旧口径**（当时契约只到 §1.5）。
> 契约此后新增 **§1.6–§1.10**（内存句柄 / 设备上下文 / 契约不变式 / 流优先级 / 流所有权与释放），
> 39 项**未覆盖新章**（台账 **E1**）。**2026-10-08 已按用户裁定把口径扩到 78 项**
> （新增 H 8 · I 10 · J 4 · K 8 · L 6 · M 3），并在 910C / P800 上完成复跑
> （**910C 73/0/5 · P800 67/0/11**，非空转 **34/5/0 · 31/8/0**）。

**✅ 2026-10-08 已按 78 项口径跑完**（网络恢复后的补齐轮 m1）⇒ **现行结论以本节为准**：

| 项 | 本轮实测（2026-10-08） |
|---|---|
| 职责审计 | **`DUTY_RESPONSE_PASS` OK 61 / FAIL 0 / SKIP 17（共 78 项）** |
| 非空转验证 | **`SELFCHECK_DUTY_EXT_PASS`：抓到 25 · 本机不适用 14 · 未抓到 0** |
| 优先级 API（台账 D1） | **`STREAM_PRIORITY_API_PASS` 7/7** ⇒ **「能设置」在新接线下成立**（三家唯一） |
| 流所有权/释放 | `STREAM_RELEASE_CONTROL_PASS`（R2/R3c/R3d 如实 SKIP，见下） |
| 报告 | `CAMBRICON_MLU_M1_RERUN_20261008.md` |

⚠️ **17 项 SKIP 均为「如实不具备」**，逐条原因见审计日志；其中三处值得单独说明：
- **J3** —— 本实例**未声明** `memory_alloc` 与 `context_lifecycle` ⇒ I3 那一族**整体不适用**。
  ⭐ 这一条**顺带暴露了一处口径缺陷**（缺陷台账**第 31 条**）：J 域原先把被委托方
  `contract_invariants` 的「不适用」（按该模块约定返回 `(True, "不适用：…")`）**压成了 `OK`**
  ⇒ 同一份审计里**同一原因两种标签**（H2/I3 报 SKIP，J3 却报 OK）。**MLU590 是首个两项都不声明的实例**，
  故此前 910C/P800 从未暴露。已按三值翻译修复。
- **L5**（「已释放对象必须明确报错」）—— ⚠️ **原预判被证伪**（见下表的更正）。

下表是**跑之前**的预判，**保留作记录**（其中 **L5 一行已实机证伪**，K5/K6/K8 与 I 系的预判成立）：

| 新域 | 本实例的预期情形（**待实测，不得先行写入结论**） |
|---|---|
| K5 / K6 / K8 | 会**真正行使**（另两家这两项是 SKIP）：区间 `(0,-3)` **非单点** ⇒ K6 必须做到「请求 `-3`/`0` ⇒ 回读一致」；K8 因非单点 ⇒ 如实 SKIP |
| ~~**L5**~~ ⚠️ **已证伪** | ❌ **实测为 `[SKIP]`**：`本机该路径未产生「本层拥有」的流（release=False）`。原因：L5 的**充分条件**不只是"区间非单点"，还要求后端存在「**厂商 C API 建流 + 本层包装**」的路径；MLU590 建流走 **torch 侧**（`torch.mlu.Stream(priority=…)`，本家**没有**独立句柄式 C API）⇒ 流归 torch 所有 ⇒ `release_stream` 必须 no-op。⇒ **三家现役实例目前都行使不了 L5**（已登记**台账 E2**） |
| I2–I5 / I10 | 取决于是否声明 `context_lifecycle`（本实例当前**未声明** ⇒ 预期如实 SKIP） |
| I1 / I4 / I5 / I10 / J3 | 属**侵入项**：审计会**自动把它们放进子进程**执行（`INVASIVE_SIDS`）。**不要**把它们改回同进程 —— 实测同进程内做上下文 create/destroy 会让后续判据被污染（910C 上表现为 I8 误报 FAIL、第二次 `check_i3` 段错误 rc=139） |

工作包见 `MLU590_FIX_WORKPACK_20261008.md`。

## 2. 本实例的运行条件（结论只在此条件下成立）

| 项 | 值 |
|---|---|
| 硬件 | 8 × MLU590-M9（98304 MiB/卡，驱动 v6.2.29 / 固件 v1.5.0） |
| 容器 | `dc-mlu590-hliu553`（`flagos-runtime-cambricon-neuware4.4.3:2.2.0`） |
| 解释器 | 容器内 `/flagos/bin/python3`（py3.10.20 / torch 2.7.1+cpu / torch_mlu 1.29.2） |
| 用卡 | `MLU_VISIBLE_DEVICES=0` |
| 设备命名空间 | `mlu`（`device_type`，**按命名空间而非厂商**） |

---

## 3. 逐项结果

| 域 | 项 | 审计点 | 结果 | 实测依据 |
|---|---|---|---|---|
| A 后端选择 | `A1` | discover() 能发现本后端 | ✅ OK | discover → ['ascend', 'cambricon', 'kunlun'] |
| A 后端选择 | `A2` | available() 含本后端 | ✅ OK | available → ['ascend', 'cambricon', 'kunlun'] |
| A 后端选择 | `A3` | use() 返回实例 / current() 一致 / 单例 | ✅ OK | current().name=cambricon, 单例=True |
| A 后端选择 | `A4` | use(未注册名) 如实抛 BackendNotFound（负向） | ✅ OK | 抛 BackendNotFound |
| B 设备域 | `B1` | device_count() > 0 | ✅ OK | n=1 |
| B 设备域 | `B2` | set_device(0) 生效且无异常 | ✅ OK | 已绑定设备 0 |
| B 设备域 | `B3` | memory_stats(0) 结构含 total/used/free | ✅ OK | keys=['free_mb', 'total_mb', 'used_mb'] |
| B 设备域 | `B4` | memory_stats 值自洽（total≈used+free 且 total>0） | ✅ OK | total=97057 used=160 free=96897 gap=0.00% |
| B 设备域 | `B5` | probe_device(0) 返回 True（健康设备） | ✅ OK | probe_device → True |
| C 流与事件 | `C1` | create_stream() 返回统一 Stream 且绑定本后端 | ✅ OK | Stream backend=cambricon |
| C 流与事件 | `C2` | create_event() 返回统一 Event 且绑定本后端 | ✅ OK | Event backend=cambricon |
| C 流与事件 | `C3` | current_stream() 可调用 | ✅ OK | current_stream 返回空 |
| C 流与事件 | `C4` | stream.synchronize() 无界路径正常返回 | ✅ OK | 无界同步返回 |
| C 流与事件 | `C5` | stream.synchronize(timeout_ms) 有界成功路径 | ✅ OK | 有界同步（流上已有工作）返回 |
| C 流与事件 | `C6` | stream.context() 可作 with 上下文管理器 | ✅ OK | with 语法可用 |
| C 流与事件 | `C7` | 跨流依赖：A record → B wait → B 可见 A 结果（真实语义） | ✅ OK | 跨流可见性结果=196609.0 |
| C 流与事件 | `C8` | stream.wait_stream(other) 可调用 | ✅ OK | wait_stream 后同步完成 |
| C 流与事件 | `C9` | event.record() 后 query() 为 True | ✅ OK | query()=True |
| C 流与事件 | `C10` | 未 record 的 event：query()=False 且 wait_host 不永久阻塞 | ✅ OK | query=False wait_host=False 耗时=0.50s |
| C 流与事件 | `C11` | record 后 event.wait_host(timeout_ms) 返回 True | ✅ OK | wait_host → True |
| C 流与事件 | `C12` | event.wait(stream) / event.synchronize() 可调用 | ✅ OK | event.wait / event.synchronize 均返回 |
| C 流与事件 | `C13` | event.elapsed_time(end) 可调用（若声明支持） | ⏹ SKIP | 未声明/不可用：RuntimeError: CNRT error: failed to call the driver-api function. |
| C 流与事件 | `C14` | 硬纪律 1：stream.record_stream(tensor) 可调用 | ✅ OK | record_stream 已登记（跨流缓冲保护） |
| D 错误翻译 | `D1` | translate_error 返回统一 FlagosError | ✅ OK | FlagosError |
| D 错误翻译 | `D2` | category 属于 L1–L4 四类 | ✅ OK | category=ErrorCategory.L1_RESOURCE |
| D 错误翻译 | `D3` | disposition 属于四动作之一 | ✅ OK | disposition='retry' |
| D 错误翻译 | `D4` | mapped / graded_by 可观测且自洽 | ✅ OK | mapped=False graded_by=default |
| D 错误翻译 | `D5` | L1–L4 处置映射符合处置约定表 | ⏹ SKIP | 本后端未声明 error_map（无厂商码样例）⇒ 交由 conformance F1 覆盖 |
| D 错误翻译 | `D6` | 完整消息不被截断（截断会退化为保守误判） | ✅ OK | category=ErrorCategory.L2_PARAM disposition=raise |
| E 状态恢复 | `E1` | device_state(0) 返回四态之一 | ✅ OK | state=available |
| E 状态恢复 | `E2` | recover_device(0,'probe') 返回 dict | ✅ OK | type=dict |
| E 状态恢复 | `E3` | 返回契约字段齐全 {ordinal,mode,recovered,state,detail} | ✅ OK | 缺字段 [] / 实际 ['detail', 'mode', 'ordinal', 'recovered', 'state'] |
| E 状态恢复 | `E4` | recovered 语义 = 设备当前可用（与 probe_device 一致） | ✅ OK | recovered=True probe_device=True |
| E 状态恢复 | `E5` | recover_device(0,'hybrid') 返回 dict | ✅ OK | mode=hybrid |
| E 状态恢复 | `E6` | recover_device(0,'real') 如实响应（执行或明确拒绝，不静默） | ✅ OK | mode=real recovered=True |
| F 硬纪律 | `F1` | 错误隔离分层：芯片级 → device_recovery；API 级 → 非 device_recovery | ⏹ SKIP | 本后端无数字错误码（未声明 error_map）⇒ 交由错误闭环与 conformance F1 覆盖 |
| G 元信息 | `G1` | supports() 与 info()['supports'] 一致 | ✅ OK | 13 项一致 |
| G 元信息 | `G2` | info()['supports'] 键集合 == _CAPABILITY_KEYS（防键漂移） | ✅ OK | KEYS-INFO 差 [] |
| G 元信息 | `G3` | known_issues() 结构完整（无则如实为空） | ✅ OK | 4 条结构完整 |

---

## 4. SKIP 项逐条说明（如实不具备，非缺口）

- **`C13` event.elapsed_time(end) 可调用（若声明支持）**：未声明/不可用：RuntimeError: CNRT error: failed to call the driver-api function.
- **`D5` L1–L4 处置映射符合处置约定表**：本后端未声明 error_map（无厂商码样例）⇒ 交由 conformance F1 覆盖
- **`F1` 错误隔离分层：芯片级 → device_recovery；API 级 → 非 device_recovery**：本后端无数字错误码（未声明 error_map）⇒ 交由错误闭环与 conformance F1 覆盖

---

## 5. 与本次三处补做的关系

本实例在 09-28 的职责响应审计中被发现/受益的补做：

- **`recover_device` 返回补 `state`**（契约五键）—— 与 `kunlun` 同款缺口（静态即确认，本次真机验证生效：
  `E3` 判 PASS）。
- `_CAPABILITY_KEYS` 补齐 `sync_timeout` 别名；删除与基类同款的 `supports()` 覆写。
- 离线自检判据数 39 → **41**（+2 条防漂移判据，已做非空转验证）。
- 本实例的补做**先在离线自检验证、后在真机复跑确认**（网络恢复后一次跑通）。

---

## 6. 复跑命令（换芯片只改 `--backend`）

```bash
# 真机（本实例；910C 需先按《名额纪律》清出带卡容器）
MLU_VISIBLE_DEVICES=0 python3 scripts/duty_response_audit.py --backend cambricon \
        --out ../MLU590/probes/duty_audit_cambricon_$(date +%Y%m%d).json

# 离线（无设备，含防漂移判据）
python3 scripts/backend_offline_check.py --backend cambricon
python3 scripts/backend_offline_check.py --all
```

**边界**：结论仅在 §2 的条件下成立；离线自检只验契约形态，**不得当本实例的真机结论**。
