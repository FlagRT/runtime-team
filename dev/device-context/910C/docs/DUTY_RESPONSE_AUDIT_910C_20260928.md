# 职责响应审计 · 910C（`ascend`）—— 39 项 sub-part 逐项实测

> **日期**：2026-09-28 ｜ **维护**：device-context（子方向 1）
> **判据来源**：《运行时层接口约定 · 设备上下文章节》`../../prototype/docs/INTERFACE_CONTRACT_DC_20260908.md`
> （§1.1 后端选择 / §1.2 设备 / §1.3 流与事件 / §1.4 错误翻译 / §1.5 状态恢复 / §2 三支撑方法 / §3 两条硬纪律）
> **工具**：`../../prototype/scripts/duty_response_audit.py` ｜ **三实例汇总报告**：`../../prototype/docs/PROTOTYPE_DUTY_RESPONSE_AUDIT_20260928.md`
> **本实例原始证据**：`../probes/duty_audit_ascend_20260928.json`

---

## 1. 结论

**`DUTY_RESPONSE_PASS`：OK 39 / FAIL 0 / SKIP 0（共 39 项）** —— **零 FAIL**。

| 判定 | 含义 | 本实例 |
|---|---|---|
| `OK` | 调用成功**且**返回值/副作用符合接口约定 | **39** |
| `FAIL` | 不响应，或响应但不符合契约（字段缺失、静默降级） | **0** |
| `SKIP` | 该能力**被本后端如实声明为不具备**，且调用被主动拦截 | **0**（不是缺口） |

---

## 2. 本实例的运行条件（结论只在此条件下成立）

| 项 | 值 |
|---|---|
| 硬件 | 16 × Ascend910（65536 MiB/卡） |
| 容器 | `flagos-infer-910c`（`quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`） |
| 解释器 | 容器内 `/usr/local/python3.11.15/bin/python3`（torch 2.11.0 + torch_npu 2.11.0） |
| 用卡 | `DEV=4`（npu:4） |
| 设备命名空间 | `npu`（`device_type`，**按命名空间而非厂商**） |

---

## 3. 逐项结果

| 域 | 项 | 审计点 | 结果 | 实测依据 |
|---|---|---|---|---|
| A 后端选择 | `A1` | discover() 能发现本后端 | ✅ OK | discover → ['ascend', 'cambricon', 'kunlun'] |
| A 后端选择 | `A2` | available() 含本后端 | ✅ OK | available → ['ascend', 'cambricon', 'kunlun'] |
| A 后端选择 | `A3` | use() 返回实例 / current() 一致 / 单例 | ✅ OK | current().name=ascend, 单例=True |
| A 后端选择 | `A4` | use(未注册名) 如实抛 BackendNotFound（负向） | ✅ OK | 抛 BackendNotFound |
| B 设备域 | `B1` | device_count() > 0 | ✅ OK | n=16 |
| B 设备域 | `B2` | set_device(0) 生效且无异常 | ✅ OK | 已绑定设备 0 |
| B 设备域 | `B3` | memory_stats(0) 结构含 total/used/free | ✅ OK | keys=['free_mb', 'total_mb', 'used_mb'] |
| B 设备域 | `B4` | memory_stats 值自洽（total≈used+free 且 total>0） | ✅ OK | total=62740 used=384 free=62355 gap=0.00% |
| B 设备域 | `B5` | probe_device(0) 返回 True（健康设备） | ✅ OK | probe_device → True |
| C 流与事件 | `C1` | create_stream() 返回统一 Stream 且绑定本后端 | ✅ OK | Stream backend=ascend |
| C 流与事件 | `C2` | create_event() 返回统一 Event 且绑定本后端 | ✅ OK | Event backend=ascend |
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
| C 流与事件 | `C13` | event.elapsed_time(end) 可调用（若声明支持） | ✅ OK | elapsed_time=0.0 |
| C 流与事件 | `C14` | 硬纪律 1：stream.record_stream(tensor) 可调用 | ✅ OK | record_stream 已登记（跨流缓冲保护） |
| D 错误翻译 | `D1` | translate_error 返回统一 FlagosError | ✅ OK | FlagosError |
| D 错误翻译 | `D2` | category 属于 L1–L4 四类 | ✅ OK | category=ErrorCategory.L1_RESOURCE |
| D 错误翻译 | `D3` | disposition 属于四动作之一 | ✅ OK | disposition='retry' |
| D 错误翻译 | `D4` | mapped / graded_by 可观测且自洽 | ✅ OK | mapped=False graded_by=default |
| D 错误翻译 | `D5` | L1–L4 处置映射符合处置约定表 | ✅ OK | 采样 4 条全部符合约定表 |
| D 错误翻译 | `D6` | 完整消息不被截断（截断会退化为保守误判） | ✅ OK | category=ErrorCategory.L2_PARAM disposition=raise |
| E 状态恢复 | `E1` | device_state(0) 返回四态之一 | ✅ OK | state=available |
| E 状态恢复 | `E2` | recover_device(0,'probe') 返回 dict | ✅ OK | type=dict |
| E 状态恢复 | `E3` | 返回契约字段齐全 {ordinal,mode,recovered,state,detail} | ✅ OK | 缺字段 [] / 实际 ['detail', 'mode', 'ordinal', 'recovered', 'state'] |
| E 状态恢复 | `E4` | recovered 语义 = 设备当前可用（与 probe_device 一致） | ✅ OK | recovered=True probe_device=True |
| E 状态恢复 | `E5` | recover_device(0,'hybrid') 返回 dict | ✅ OK | mode=hybrid |
| E 状态恢复 | `E6` | recover_device(0,'real') 如实响应（执行或明确拒绝，不静默） | ✅ OK | mode=real recovered=True |
| F 硬纪律 | `F1` | 错误隔离分层：芯片级 → device_recovery；API 级 → 非 device_recovery | ✅ OK | 芯片级=device_recovery API级=raise |
| G 元信息 | `G1` | supports() 与 info()['supports'] 一致 | ✅ OK | 13 项一致 |
| G 元信息 | `G2` | info()['supports'] 键集合 == _CAPABILITY_KEYS（防键漂移） | ✅ OK | KEYS-INFO 差 [] |
| G 元信息 | `G3` | known_issues() 结构完整（无则如实为空） | ✅ OK | 1 条结构完整 |

---

## 4. SKIP 项逐条说明（如实不具备，非缺口）

- 本实例**无 SKIP 项**（39 项全部 OK）。

---

## 5. 与本次三处补做的关系

本实例在 09-28 的职责响应审计中被发现/受益的补做：

- 三处补做中，**本实例原本就正确**：`recover_device` 早已返回契约五键（`state` 齐全），
  因此它是另两家的**对照基准**（正是"ascend 有、另两家没有"这一不对称暴露了缺口）。
- 无代码改动落到本实例；补做后**回归复跑仍 39/0/0**（无退化）。

---

## 6. 复跑命令（换芯片只改 `--backend`）

```bash
# 真机（本实例；910C 需先按《名额纪律》清出带卡容器）
DEV=4 python3 scripts/duty_response_audit.py --backend ascend \
        --out ../910C/probes/duty_audit_ascend_$(date +%Y%m%d).json

# 离线（无设备，含防漂移判据）
python3 scripts/backend_offline_check.py --backend ascend
python3 scripts/backend_offline_check.py --all
```

**边界**：结论仅在 §2 的条件下成立；离线自检只验契约形态，**不得当本实例的真机结论**。
