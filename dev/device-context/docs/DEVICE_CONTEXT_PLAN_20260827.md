# 设备执行上下文（device-context）方案定稿

> 状态：✅ 定稿（2026-08-27）｜ 作者：Kistich ｜ 适用范围：运行时组设备执行上下文方向（细项21）
> 对应看板任务 #6（完整方案文档 + PR 提交）——本文档为 PR 合入 dev-1.0 的方案正文

---

## 1. 目标与边界

**设备执行上下文**负责统一管理不同芯片设备的初始化、上下文创建、执行队列、Stream/Event、设备间同步、Host 与 Device 数据传输、错误捕获和状态恢复。

对于昇腾、寒武纪、昆仑芯、燧原、壁仞、平头哥等不同芯片，运行时层通过 **Backend 插件**封装厂商 Runtime 接口，向上提供统一**设备句柄、内存句柄和执行句柄**。该机制保证同一模型部署产物在不同芯片上能够通过一致的调用方式运行，降低上层模型服务和运维系统的适配复杂度。

| 需求 | 内容 | 落地位置 |
|------|------|----------|
| 需求 1 | 封装不同芯片 Runtime：统一设备/内存/执行句柄/生命周期 | Backend 插件层（厂商 Runtime 封装） |
| 需求 2 | 统一 Stream 语义/异步传输/页锁定/双缓冲/同步/错误码翻译/状态恢复 | 统一事件/错误/状态适配层 |
| 需求 3+4 | 张量并行/流水线并行下的跨卡同步与执行编排 | FlagCX（通信）+ 训练侧验证 |

---

## 2. 设计

### 2.1 分层架构

```
┌─────────────────────────────────────────────────┐
│ 上层：模型服务 / 训练支撑 / 运维系统（芯片无关）      │
├─────────────────────────────────────────────────┤
│ 设备执行上下文（本方向）                          │
│  ├─ 统一设备句柄（枚举/初始化/生命周期）           │
│  ├─ 统一内存句柄（分配/释放/页锁定/双缓冲）        │
│  ├─ 统一执行句柄（Stream/Event/异步传输/同步）     │
│  ├─ 统一错误对象（错误码翻译 L1-L4 三维投影）      │
│  └─ 设备状态机 + 五段式恢复                      │
├─────────────────────────────────────────────────┤
│ Backend 插件（封装厂商 Runtime）                  │
│  └─ torch_npu / torch_mlu / ... 统一语义适配      │
└─────────────────────────────────────────────────┘
```

### 2.2 统一事件语义契约（E 系列）

- **E1**：`event.record(stream)` / `stream.wait(event)` —— 设备流等待语义
- **E2**：`wait_host(timeout_ms)` 主机有界等待——**永不永久阻塞**，超时返回 False（逃生主路径为 `query`）
- **E3**：未 record 事件 `query` 返回未完成（修复 ACL/原生 Event 默认完成状态误报）
- **E4**：`recorded` 状态跟踪（事件生命周期内显式记录）

> 适配层 `npu_events.py` 补齐 torch_npu 原生 Event 缺口：未 record query 误报 → recorded 跟踪修正；无 wait_host → 轮询实现。

### 2.3 统一错误对象（F 系列）

- `ErrorCategory` L1-L4（参数/资源/执行/系统）
- `FlagosError` 三维投影：`category` / `location` / `root_cause` + `error_code` + `is_retryable` / `is_fatal`
- `translate_error`：错误码优先 → 消息粗分类 → L3 默认
- 兼容两形态：B 线 `ret=161002` 与 A 线 `error code is 161002`

### 2.4 状态机与恢复（R 系列）

- **DeviceState 四态**：AVAILABLE / DEGRADED / ISOLATED / DESTROYED（状态注册表 + 查询/订阅/转换事件 + 快照）
- **五段式恢复**：captured → evaluated → isolated → recovered → replay_ready
  - R1 `handle_error`：捕获编排（L1-L3 不触发设备恢复）
  - R2 `probe_device` / `evaluate_device`：评估
  - R3/R4 `recover_device`：仅 ISOLATED 可重建（探针重试 ≤3 次）
  - R5 `mark_inflight` / `finish_inflight` / `replay_tasks`：在途登记 = 重放集合数据源

---

## 3. 昇腾 910C 完成度（A 线，torch_npu 2.10.0）

| 职责项 | 验证状态 | 证据 |
|---|---|---|
| 设备初始化/上下文/统一句柄 | ✅ | conformance 设备无关化（--backend npu/flagos），新容器复现 |
| 执行队列 / Stream / Event | ✅ | S1-S4 + E1/E2/E3 + npu_events 适配层 |
| 设备间同步（跨卡通信） | ✅ | 同构 2 卡 flagcx 训练：allgather [1,2]/[10,11] ✓、DDP 全程稳定 |
| Host↔Device 传输（页锁定/双缓冲） | ✅ | T1/T2/T3 + 六探针 6/6 |
| 错误捕获 / 状态恢复 | ✅ | F1 + R 恢复 + errors 三维翻译 |
| 端到端训练验证 | ✅ | Qwen2.5-1.5B 双卡 DDP 2481 步全程：loss 1.9501 / 4245 tok/s（flagcx）、5428 tok/s（hccl） |

- **conformance 13/13 CONFORMANCE_PASS**（S1/S2/S3/S4/E1/E2v2/E3/T1/T2/T3/F1/R1-R5，results_aline_20260826/）
- **六探针 6/6 PASS**（错误翻译/拓扑/锁页池/双缓冲/恢复/CPU-NPU）
- **FlagCX 插件安装方法沉淀**（flagcx_plugin_setup.md + setup_flagcx_plugin.sh，幂等可用）

---

## 4. 如实标注的缺口（不阻塞换芯片，但需在新芯片/后续迭代中关注）

| # | 缺口 | 现状 | 影响与后续 |
|---|---|---|---|
| 1 | **T2 引用计数观测** | 框架层无法观测张量引用计数，标注"依赖运行时登记表" | 在途拷贝保护依赖登记表实现，需设备生命周期接口配合 |
| 2 | **T3 统一拓扑查询** | torch_npu 未暴露统一拓扑接口 | 拓扑事实经 npu-smi/外部通道获取；新芯片需厂商 API 对齐 |
| 3 | **E2 wait 未 record 边界** | 原生 Event `wait` 未 record 可能永久阻塞（无超时） | 统一层已提供 `wait_host` 有界等待收敛；新芯片用例需覆盖验证 |
| 4 | **状态恢复重建近似** | 重建为框架层最小近似（探针重试=重取资源验证） | 真实上下文重建待 torch_fl/厂商设备生命周期接口 |
| 5 | **FlagCX 吞吐差距（挂账）** | flagcx 4157 vs hccl 5428 tok/s；per-call CPU 开销 0.72ms vs 0.02ms（36 倍）、总 CPU 耗时 18 倍；已排除 coalesced 嫌疑 | 修复方向 = FlagCX adaptor per-call 路径精简（事件/流管理热路径）；**挂账移交通信方向协作评估**，非本方向阻塞项 |

---

## 5. 下一芯片接入方法

conformance 框架为**设备无关**设计，接入新芯片只需三步：

1. 同套用例跑新芯片：`python runner.py --backend <npu|flagos|mlu|...>`，比对 `ok` 字段——**行为差异即缺陷**
2. 六探针脚本复用：错误翻译 / 拓扑 / 锁页池 / 双缓冲 / 恢复 / CPU-NPU，逐项比对
3. 统一事件/错误适配层按厂商 API 补充（参照 npu_events.py / errors.py 模式）

候选芯片（FlagCX 官方已支持、通信零开发量）：寒武纪（CNCL，torch_mlu）、昆仑芯（XCCL，xpytorch）。

---

## 6. 交付物清单（PR 范围）

| 类别 | 内容 |
|---|---|
| 方案文档 | 本文档 + event_semantics_contract.md |
| 测试资产 | conformance 框架（runner/cases/device_state/errors/npu_events/recovery）+ 六探针 + 结果 JSON |
| FlagCX 缺陷修复 | P2/P6/P7 复现补丁 + FLAGCX_CORE_DEFECT_FIXES_20260826.md |
| 看板 | README.md（任务看板 + 批次记录） |
