# 2026.09 最小通信 Backend 契约与验收矩阵

> 负责人：尤联忠 ｜ 状态：接口草案 v0.2 ｜ 更新：2026-09-09

## 1. 交付目标与边界

当前统一目标是在 `dev/stack.lock.910c.v1.yaml` 锁定的训练镜像内支撑 Qwen3-Embedding-0.6B 双卡微调，并提供可追溯的通信正确性对照。本方向把既有 FlagCX 能力接入统一执行链路，不负责模型切分策略，不重复实现 Stream/Event、显存分配或厂商通信协议。

首轮只建立小规模、确定性的正确性基线；正确性闭环后再扩大卡数、消息规模并开展通信计算重叠优化。

## 2. v0 最小语义契约

| 能力域 | 最小能力 | 语义要求 | 首轮验收 |
| --- | --- | --- | --- |
| 通信组 | 创建、查询、销毁 | Rank 与 world size 唯一；重复销毁安全；异常退出可清理 | 2 Rank 创建/销毁，无残留进程 |
| Collective | AllReduce、AllGather、ReduceScatter、Broadcast | 明确输入输出形状、dtype、原地/非原地支持；结果确定 | FP32/BF16 小张量逐元素比对 |
| 点对点 | Send、Recv | 明确 src/dst、tag/序号和完成条件 | Rank 0→1、1→0 双向校验 |
| 异步完成 | `async_op`、`wait`、`is_completed` | 返回只表示已提交；消费前由 Work/Event 建立依赖，禁止把 Host 返回误当设备完成 | 立即查询、显式等待、下游消费三类用例 |
| Stream/Event | 调用者当前流与跨流依赖 | Backend 不私自切换全局默认流；跨流消费必须有可观察依赖 | 当前流、双流竞争、同步后结果一致 |
| 能力声明 | Backend、设备、操作、dtype、拓扑限制 | 不支持的组合在执行前显式拒绝，禁止静默降级 | 能力表与实际探针一致 |
| 错误与恢复 | 初始化、超时、设备/通信失败 | 错误可分类、可追踪；通信组进入失败态后不继续提交新任务 | 至少覆盖初始化失败与超时路径 |

说明：具体 C++/Python API 名称以 FlagCX 和 Torch distributed 现有接口为准，本文件冻结的是运行时所需语义，不另造平行 API。

## 3. 首轮验收矩阵

| 层级 | 场景 | 规模 | 通过标准 | 结果位置 |
| --- | --- | --- | --- | --- |
| L0 环境 | FlagCX 导入、Backend 注册、设备可见 | 单进程 | 版本、commit、设备和环境变量完整记录 | `docs/results/<date>/env.md` |
| L1 通信组 | init/destroy | 2 Rank | 两端初始化、销毁成功，无残留进程 | `docs/results/<date>/group.json` |
| L2 正确性 | 基础 Collective | 2 卡 | 各操作 FP32/BF16 结果与 Host 参考逐元素一致 | `docs/results/<date>/collectives.json` |
| L3 完成语义 | 同步/异步、当前流/跨流消费 | 2 卡 | `wait`、Event 和下游消费行为符合契约 | `docs/results/<date>/async.json` |
| L4 稳定性 | 循环 Collective | 2 卡，100 轮 | 无超时、NaN、数据漂移或残留进程 | `docs/results/<date>/stability.json` |
| L5 扩展基线 | Collective 带宽/时延 | 16 卡 | 固定消息大小、预热和迭代，归档原始数据 | `benchmarks/results/<date>/` |

当前阶段先完成双卡正确性、训练腿闭环和接口约定。更大卡数与性能测试属于后续规模化验证，不作为本轮双卡原型的合入前置条件。

### 当前执行结果

- 新增统一探针 `dev/communication/probes/communication_correctness.py`，覆盖 AllReduce、AllGather、P2P 和异步 AllReduce。
- 锁定训练镜像内使用 2 Rank、FP32/BF16、20 轮执行，共 320 次通信调用，结果 320/320 PASS、0 失败。
- 异步 AllReduce 提交后立即查询 `Work.is_completed()`，共 80/80 次返回 `true`；最终结果在 `wait` 与设备同步后正确。该现象已列为后续跨流完成语义验证项。
- 执行记录与原始 JSON：`docs/COMMUNICATION_CORRECTNESS_20260909.md`、`results/20260909/`。

## 4. 已对齐的可复用资产

- 双卡 FlagCX AllReduce 冒烟：`dev/memory/probes/flagcx_smoke.py`。该脚本已显式记录 FlagCX 异步返回后需设备同步的现状。
- Route A 多尺寸 AllReduce：`dev/memory/probes/routeA_s2_3_allreduce.py`，可参考其正确性和带宽记录方式，但 P800 环境不能直接作为 910C 结论。
- Work 完成语义：`dev/device-context/distributed_training/scripts/test_work_sem.py`。
- TP 通信与跨流验证：`dev/device-context/distributed_inference/inference/test_tp_comm_sync_enhanced.py`。
- 设备上下文方向已在 `dev-1.0` 归档 Qwen3-4B TP=1/2/4 greedy 输出一致性结果，可作为上层链路证据；本方向仍需独立完成基础 Collective 的可重复基线。

复用方式：通信方向保留统一执行入口和结果索引；源探针仍由原目录维护，避免复制后发生版本漂移。若探针需要通用化，先在原目录提取公共参数，再由双方共同确认。

## 5. 执行顺序与责任接口

1. 固定 FlagCX、Torch/设备插件、CANN、驱动及镜像版本，记录 commit 与环境快照。
2. 使用通信方向统一探针核实 Backend 注册、通信组生命周期，以及 AllReduce、AllGather、P2P 的确定性结果。
3. 在当前三类对照稳定后，再补 ReduceScatter、Broadcast 等操作，不扩大本轮双卡原型的强制范围。
4. 与设备上下文方向共同确认当前流、Event 和 Work 完成语义，先消除“Host 返回即设备完成”的歧义。
5. 与设备方向共同确认训练腿证据，满足组内合入把关后自行合入 `dev-1.0`；后续再执行 8 卡及以上规模验证。

## 6. 当前风险与降级

- 共享 910C 服务器带卡容器并发上限为 3；两条腿按 `stack.lock` 要求串行执行，规模化性能验证另行预约资源窗口。
- 通信验证资产分散在 `device-context` 与 `memory`；以本文件作为通信验收索引，避免重复维护实现副本。
- FlagCX 的异步完成、当前流和 Backend 适配语义曾出现差异；所有上层 TP/PP 验证前必须先通过 L3，不以单次模型跑通替代基础语义验收。
- 若某 Collective 暂不支持，能力声明应显式标记并由上层选择安全路径；不得静默返回成功或伪造同步完成。
