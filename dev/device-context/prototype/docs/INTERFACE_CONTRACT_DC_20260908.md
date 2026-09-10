# 运行时层接口约定 · 设备上下文章节
## —— flagos-runtime v0.1.0（2026-09-08 定稿）

> **读者**：运行时层各子方向（显存/分布式/监控/精度/算子/调度）及上层算子层、编译层的对接人。
> **效力**：本文档是设备上下文方向的**接口承诺**；下游按本文档接入，接口行为以本文档为准。
> **配套**：组件说明 `runtime/README.md` ｜ 基座配置 `dev/stack.lock.910c.v1.yaml`

---

## 1. 统一 API 承诺（下游直接使用）

### 1.1 后端选择

```python
runtime.use("ascend")     # 选择后端；换芯片只改这一行
runtime.available()       # ["ascend"] —— 当前可用后端
runtime.discover()        # 扫描已安装的 backend 插件
```

- 语义：`use()` 之后，全部设备操作路由到所选后端；**同一份业务代码不因换芯片而修改**
- 约束：`use()` 必须在任何设备操作之前调用；重复调用视为切换（需重新 set_device）

### 1.2 设备

| 接口 | 签名 | 语义 |
|---|---|---|
| `device_count()` | `→ int` | 可见设备数 |
| `set_device(ordinal)` | `ordinal: int` | 绑定当前设备；后续操作默认在此设备 |
| `memory_stats()` | `→ dict` | 显存统计（total/used/峰值等字段，显存方向经此采集，口径以此为准） |
| `probe_device(ordinal)` | `→ bool` | 设备探活（轻量，不干扰业务） |

### 1.3 流与事件

| 接口 | 语义 | 关键约束 |
|---|---|---|
| `create_stream()` | 创建统一 Stream 对象 | 跨流传缓冲必须 `record_stream`（见 §3 纪律 1） |
| `create_event()` | 创建统一 Event 对象 | record 后 query 才有意义 |
| `stream.wait_event(ev)` | 建立跨流依赖（A record → B wait → B 可见 A 的结果） | 事件必须先 record |
| `stream.synchronize(timeout_ms)` | **有界**流同步；超时抛 `TimeoutError` | 长驻服务必须用有界同步，防整体 hang |
| `event.wait_host(timeout_ms)` | **有界**主机侧等待 | 同上 |
| `stream_ctx` | 流上下文（with 语法） | 上下文内的操作入该流 |

### 1.4 错误翻译

```python
fe = runtime.translate_error(exc, location="...")
# fe.category      → L1_RESOURCE / L2_PARAM / L3_EXECUTION / L4_FATAL
# fe.disposition   → retry / raise / replay / device_recovery
# fe.mapped        → 是否命中已知错误码映射
# fe.graded_by     → 分级来源（code_map=映射表 / message_hint=消息规则）
```

**处置约定**（下游必须按 disposition 处理，禁止按错误消息字符串自行判断）：

| 分级 | 含义 | 处置 | 责任方 |
|---|---|---|---|
| L1_RESOURCE | 资源类（显存不足等） | 重试或交显存方向扩容 | 显存/设备 |
| L2_PARAM | 参数类 | **上抛调用方**（重试无意义） | 调用方 |
| L3_EXECUTION | 执行类（如流同步超时 507046） | 重放（有机会成功）+ 调度重排 | 设备+调度 |
| L4_FATAL | 芯片级致命（如 AICORE 异常 507015） | **设备级恢复** `recover_device` + 检查点恢复 | 设备+监控+分布式 |

### 1.5 状态恢复

| 接口 | 语义 |
|---|---|
| `device_state(ordinal)` | 设备四态查询（AVAILABLE / DEGRADED / ISOLATED / UNKNOWN） |
| `recover_device(ordinal, mode)` -> **dict** | 三级重建：`probe`（保底探活）/ `real`（CANN 官方 aclrtResetDevice 序列）/ `hybrid`（先 probe 后 real） |

- **调用约定**：监控方向做检测与恢复编排（何时调、调哪级），恢复执行由本组件完成
- **约束**：L4 级错误流级重试无效，必须走 `recover_device`；real 模式当前默认不启用
  （本地多进程联调已过，生产默认前需多卡压测调优——见 9 月计划 W4 遗留）

---

## 2. Backend 插件接入规范（新芯片方向照此实现）

**新增一家芯片 = 实现一个 backend + 跑通 conformance。** 步骤：

1. 新建 `runtime/backends/<vendor>/`，实现 `RuntimeBackend` 抽象（`backends/base.py`）：
   - **五域抽象**：设备（count/set_device/memory_stats/probe）、内存（分配/释放/统计）、
     Stream-Event（创建/等待/有界同步/上下文）、错误码翻译、状态恢复
   - 三个多流支撑方法：`stream_context / synchronize_stream（有界）/ wait_event_host（有界）`
   - 可选能力用 `supports()` 声明（conformance 自动生成 stub-skip 报告）
2. 提供 `build()` 工厂函数，在 `registry` 登记名字
3. 跑通 conformance：`python3 runtime/conformance/runner.py --backend <vendor>`
   ——13 例 + 6 例全绿（或 stub-skip 报告说明缺口）即接入完成
4. 接入成本目标：**≤5 人天**（11 月昆仑芯/寒武纪实做验证）

---

## 3. 两条硬纪律（多流正确性，违反即数据错乱）

1. **`record_stream` 跨流保护**：张量缓冲在流 A 分配、交流 B 使用时，必须
   `buffer.record_stream(stream_B)`，告知缓存分配器该缓冲仍在被使用，
   否则可能被提前回收重用 → 数据竞争。（统一 Stream 已封装，直接调用）
2. **错误隔离分层**：API 级错误（如 107015）只影响该次调用，其他流不受影响，可流级重试；
   **芯片级错误（AICORE_TIMEOUT/EXCEPTION）影响该设备全部流**，流级重试无效，
   必须走设备级 `recover_device`。

---

## 4. 版本与兼容承诺

| 版本 | 承诺 |
|---|---|
| v0.1（当前） | 允许破坏性变更（提前一周知会下游）；下游锁定 `use("ascend")` 用法不变 |
| v0.1.x | 吸收 9 月下游反馈，不改已有接口签名 |
| v0.2（10 月） | 20 模型反馈迭代；新增能力不影响已有调用 |
| v1.0（2027.06） | **稳定接口承诺**：变更需评审 |

**变更流程**：接口变更 → 本文档更新（变更记录节）→ 知会全部下游 → conformance 回归全绿。

### 变更记录

| 日期 | 版本 | 变更 | 知会 |
|---|---|---|---|
| 2026-09-08 | v0.1.0 | 初版定稿（API 面 + 插件规范 + 两条纪律） | 运行时层全组 |

---

## 5. 验收与支持

- **组件质量基线**：昇腾真机冒烟 37/37；conformance 13/13 + 6/6；跨天 28h 长驻零增长
- **下游接入自检**：接入后先跑 `python3 runtime/smoke_runtime.py`，全过即接入成功
- **问题反馈**：device-context（Kistich）；每周五前反馈的问题当周定位、下周版本修复
- **本周状态**：组件 v0.1 待打包下发（见 9 月计划 W2）；本章节即战略文档要求的
  "设备上下文接口约定章节"定稿

---

## 补充：recover_device 返回契约（2026-09-09 统一）

- **返回类型统一为 `dict`**：`{ordinal, mode, recovered, state, detail}`
- **`recovered` 语义 = 设备当前可用**（不是"是否执行了重建"）
  - 底层 `recovery.recover_device` 仅在设备处于 ISOLATED 时才执行重建，
    否则返回 False；此前 ascend 后端直接透传该 bool，导致"设备正常、无需重建"
    被上报为"恢复失败"。现已统一：以设备状态 + 探活结果判定。
- `detail` 区分三种情况：重建成功 / 无需重建（探活可用）/ 恢复失败（探活不可用）
- `state` 为设备四态之一，便于上层与监控方向判定

---

## 补充：错误分级调用纪律（2026-09-09 实测）

**调用 `translate_error` 时必须传入完整的原始异常 / 服务错误消息，不得截断。**

实测：vLLM 服务对超长输入返回 HTTP 400，错误体含
`"type":"BadRequestError","param":"input_tokens"` 与完整 message。
- 传完整 message → 正确分级 **L2_PARAM / raise**（参数类上抛，重试无意义）
- 仅传 "HTTP 400"（或错误体被截断导致 JSON 解析失败）→ 退化为 **L3_EXECUTION / replay**
  —— 语义丢失导致保守误判，会让上层对参数错误做无意义的重放。

建议：
- 服务端透传 vLLM 异常类型与 message，不要只留状态码；
- 客户端调用 `translate_error` 时把原始异常对象（或完整消息）传入，不做截断；
- 若只能拿到状态码，需显式使用状态码→分级映射，而不是交给消息兜底。
