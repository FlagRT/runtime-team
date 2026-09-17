# 统一事件语义契约（E1-E4）修订版 —— 昇腾实测驱动

> **状态**：v2（2026-08-22 修订）。修订动因：昇腾 910C 源码版 torch_fl 实测发现
> `flagos.Event().wait()` 在事件未 record 时**永久阻塞**（aclrtStreamWaitEvent 对
> 未 record 事件的主机侧同步行为），无逃生通道——违反契约的"行为确定"要求。
> 本文件为 torch_fl 实现与一致性测试用例的统一依据；与《实施方案》附录A（行为契约
> 形式化规约）的 E1-E4 条款保持一致（附录A E2 已同步修订）。

## E1 记录语义（不变）

- **前置条件**：在流 S 上调用 record(E)。
- **保证**：事件 E 标记流 S 上此前全部操作的完成点，E 置 RECORDED，待流推进至该点时置 COMPLETED。
- **违反即缺陷**：E 在流未推进至 record 点前置 COMPLETED，或 record 后 E 状态不变。
- **可观测验证点**：E 的状态转换 INIT→RECORDED→COMPLETED 与流推进一致。

## E2 等待语义（v2 修订：新增超时逃生）

- **前置条件**：消费流 S_c 调用 wait(E)；或主机侧调用 wait_host(E, timeout_ms)。
- **保证**：
  1. `wait(E)`（设备流等待）：若 E 已 COMPLETED，S_c 立即推进；若 E 已 RECORDED 未 COMPLETED，S_c 等待至 E COMPLETED；若 E 未 record，等价于等待 E 的下一次 record 并完成——语义确定，不返回、不崩溃。
  2. `wait_host(E, timeout_ms)`（主机有界等待）：轮询 E 完成状态，完成则返回 COMPLETED；超过 timeout_ms 未完成则返回 TIMEOUT——**永不永久阻塞**（timeout_ms=None 视为无界，仅应在 E 已被 record 的前提下使用）。
- **违反即缺陷**：wait(E) 在未 record 事件上行为不确定（返回、崩溃）；wait_host 无超时逃生路径导致永久阻塞；wait_host 超时后事件状态无法恢复查询。
- **可观测验证点**：先 wait 后 record 的依赖成立性测试用例；wait_host 超时返回 TIMEOUT 且后续 query(E) 可正常继续的测试用例。

## E3 查询语义（v2 强化：查询是逃生主路径）

- **前置条件**：调用 query(E)（非阻塞）。
- **保证**：返回 E 是否 COMPLETED，不阻塞调用线程；对**未 record 的事件返回未完成（不报错、不崩溃）**——query 是"先 wait 后 record"场景的主机侧逃生主路径。
- **违反即缺陷**：query 阻塞调用线程，或对未 record 事件抛出异常/崩溃，或返回与 E 实际状态不一致。
- **可观测验证点**：未 record 事件 query 返回未完成；record 并推进后 query 返回完成。

## E4 计时语义（不变）

- **前置条件**：两事件 E_start、E_end 在同流 S 上 record 且均 COMPLETED。
- **保证**：elapsed(E_start, E_end) 返回两 record 点之间的设备侧耗时（精度按能力声明档位，补偿档如实反映）。
- **违反即缺陷**：计时精度低于声明档位且未如实登记，或返回与流推进不一致的耗时。
- **可观测验证点**：计时事件流与流推进时序一致；补偿档的采样时间戳可观测。

## 实现要求（torch_fl）

| 要求 | 落点 | 说明 |
|---|---|---|
| wait(E) 语义确定 | `AclEvent.wait` | 保持设备流等待语义；未 record 事件的等待行为确定为"等待下一次 record" |
| **wait_host(timeout_ms) 新增** | `AclEvent.wait_host` + `Event.wait_host` | 主机有界等待：循环 query + 有界 sleep，完成返回 COMPLETED / 超时返回 TIMEOUT；**timeout 参数必选逃生路径** |
| query 对未 record 不崩溃 | `AclEvent.query` | 未 record 时返回未完成（依赖 aclrtQueryEventWaitStatus 的实际返回，需实测确认） |
| 统一事件接口暴露 | `flagos.Event` | Event 类增加 wait_host 方法，与 record/wait/query 平级 |

## 测试要求（conformance）

| 用例 | 验证点 | 期望 |
|---|---|---|
| e2_wait_before_record | record→wait 正常路径 | PASS |
| **e2_host_timeout（新增）** | 未 record 事件 wait_host(200ms) | 返回 TIMEOUT（≤1s），不永久阻塞；随后 record→wait 仍正常 |
| **e3_query_unrecorded（新增）** | 未 record 事件 query | 返回未完成，不崩溃 |
