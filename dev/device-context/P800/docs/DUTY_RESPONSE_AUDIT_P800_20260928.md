# 职责响应审计 · P800（`kunlun`）—— 39 项 sub-part 逐项实测

> **日期**：2026-09-28 ｜ **维护**：device-context（子方向 1）
> **判据来源**：《运行时层接口约定 · 设备上下文章节》`../../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md`
> （§1.1 后端选择 / §1.2 设备 / §1.3 流与事件 / §1.4 错误翻译 / §1.5 状态恢复 / §2 三支撑方法 / §3 两条硬纪律）
> **工具**：`../../prototype/scripts/duty_response_audit.py` ｜ **三实例汇总报告**：`../../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md`
> **本实例原始证据**：`../probes/duty_audit_kunlun_20260928.json`

---

## 1. 结论

**`DUTY_RESPONSE_PASS`：OK 36 / FAIL 0 / SKIP 3（共 39 项）** —— **零 FAIL**。

| 判定 | 含义 | 本实例 |
|---|---|---|
| `OK` | 调用成功**且**返回值/副作用符合接口约定 | **36** |
| `FAIL` | 不响应，或响应但不符合契约（字段缺失、静默降级） | **0** |
| `SKIP` | 该能力**被本后端如实声明为不具备**，且调用被主动拦截 | **3**（不是缺口） |

---

## 1.1 口径更新（2026-10-08 · 第十轮 r10 —— **扩口径后本实例的现行结论**）

> ⚠️ 本文件 §1 的「**39 项**」是 **2026-09-28 的旧口径**（当时契约只到 §1.5）。
> 契约此后新增 §1.6–§1.10，39 项**未覆盖新章**（台账 **E1**）。**2026-10-08 已按用户裁定扩口径**：

| 项 | 值 |
|---|---|
| 审计总项数 | **78 项**（新增 H 8 · I 10 · J 4 · K 8 · L 6 · M 3） |
| **本实例现行结论（2026-10-08 · HEAD 含第 30 条修复）** | **`DUTY_RESPONSE_PASS`：OK 67 / FAIL 0 / SKIP 11（共 78 项）** |
| 11 项 SKIP（**如实不具备，非缺口**） | 旧 3 项（`C13` 事件计时 · `D5`/`F1` 无数字错误码） + 新 8 项：`I2`–`I5`/`I10`（**未声明 `context_lifecycle`** ⇒ 上下文生命周期四件事不成立，只能只读观测）· `K2`/`K3`（已声明 `stream_priority` 与 `stream_priority_control` ⇒ 「未声明」分支不适用）· `L5`（单点区间走「等价放行」⇒ 不产生「由本层拥有」的流） |
| 非空转验证 | **31 抓到 / 8 本机不适用 / 0 未抓到** ⇒ `SELFCHECK_DUTY_EXT_PASS` |
| 同轮破坏面回归 | 离线 **106/0/1** · 对称性 **7/0** · 冒烟 **46/0** · conformance **13+6** · 契约不变式 **4/4** · 根解析自检 **40/0** |
| 证据 | `../probes/e1_regress_kunlun_20261008_out/e1_duty_kunlun.json`、`e1_selfcheck_duty_ext.json`、`../probes/e1_p800_20261008.log` |

## 2. 本实例的运行条件（结论只在此条件下成立）

| 项 | 值 |
|---|---|
| 硬件 | 8 × XPU（98304 MiB/卡，驱动 5.0.21.47） |
| 容器 | `hliu553-device-context-p800` |
| 解释器 | conda `python310_torch29_cuda`（torch 2.9.0+cu129） |
| 用卡 | `CUDA_VISIBLE_DEVICES=5` |
| 设备命名空间 | `cuda`（`device_type`，**按命名空间而非厂商**） |

---

## 3. 逐项结果

| 域 | 项 | 审计点 | 结果 | 实测依据 |
|---|---|---|---|---|
| A 后端选择 | `A1` | discover() 能发现本后端 | ✅ OK | discover → ['ascend', 'cambricon', 'kunlun'] |
| A 后端选择 | `A2` | available() 含本后端 | ✅ OK | available → ['ascend', 'cambricon', 'kunlun'] |
| A 后端选择 | `A3` | use() 返回实例 / current() 一致 / 单例 | ✅ OK | current().name=kunlun, 单例=True |
| A 后端选择 | `A4` | use(未注册名) 如实抛 BackendNotFound（负向） | ✅ OK | 抛 BackendNotFound |
| B 设备域 | `B1` | device_count() > 0 | ✅ OK | n=1 |
| B 设备域 | `B2` | set_device(0) 生效且无异常 | ✅ OK | 已绑定设备 0 |
| B 设备域 | `B3` | memory_stats(0) 结构含 total/used/free | ✅ OK | keys=['free_mb', 'total_mb', 'used_mb'] |
| B 设备域 | `B4` | memory_stats 值自洽（total≈used+free 且 total>0） | ✅ OK | total=98304 used=166 free=98138 gap=0.00% |
| B 设备域 | `B5` | probe_device(0) 返回 True（健康设备） | ✅ OK | probe_device → True |
| C 流与事件 | `C1` | create_stream() 返回统一 Stream 且绑定本后端 | ✅ OK | Stream backend=kunlun |
| C 流与事件 | `C2` | create_event() 返回统一 Event 且绑定本后端 | ✅ OK | Event backend=kunlun |
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
| C 流与事件 | `C13` | event.elapsed_time(end) 可调用（若声明支持） | ⏹ SKIP | 未声明/不可用：ValueError: Both events must be created with argument 'enable_timing=True'. |
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
| G 元信息 | `G3` | known_issues() 结构完整（无则如实为空） | ✅ OK | 1 条结构完整 |

---

## 4. SKIP 项逐条说明（如实不具备，非缺口）

- **`C13` event.elapsed_time(end) 可调用（若声明支持）**：未声明/不可用：ValueError: Both events must be created with argument 'enable_timing=True'.
- **`D5` L1–L4 处置映射符合处置约定表**：本后端未声明 error_map（无厂商码样例）⇒ 交由 conformance F1 覆盖
- **`F1` 错误隔离分层：芯片级 → device_recovery；API 级 → 非 device_recovery**：本后端无数字错误码（未声明 error_map）⇒ 交由错误闭环与 conformance F1 覆盖

---

## 5. 与本次三处补做的关系

本实例在 09-28 的职责响应审计中被发现/受益的补做：

- **`recover_device` 返回补 `state`**（契约五键 `{ordinal, mode, recovered, state, detail}`）——
  本实例**首轮即判 FAIL**（`缺字段 ['state']`），是本次审计暴露的第一处缺口；补做后 `E3` 转 PASS。
- `_CAPABILITY_KEYS` 补齐 `sync_timeout`（`bounded_sync` 的弃用别名）⇒ `supports()` 跨芯片判定一致。
- 删除与基类同款的 `supports()` 覆写（收敛到唯一实现，别名归一只在基类维护一处）。
- 离线自检判据数 39 → **41**（+2 条防漂移判据，已做非空转验证）。

---

## 6. 复跑命令（换芯片只改 `--backend`）

```bash
# 真机（本实例；910C 需先按《名额纪律》清出带卡容器）
CUDA_VISIBLE_DEVICES=5 python3 scripts/duty_response_audit.py --backend kunlun \
        --out ../P800/probes/duty_audit_kunlun_$(date +%Y%m%d).json

# 离线（无设备，含防漂移判据）
python3 scripts/backend_offline_check.py --backend kunlun
python3 scripts/backend_offline_check.py --all
```

**边界**：结论仅在 §2 的条件下成立；离线自检只验契约形态，**不得当本实例的真机结论**。
