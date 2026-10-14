# 调度引擎四层架构说明（sched_engine）

> 对应任务：在动态组批实现基础上构建与模型、推理框架、芯片解耦的调度引擎。
> 本轮交付：vLLM + Qwen3-Embedding-0.6B + 昇腾 910C 可运行链路（代码与脚本），
> 请求队列 / 优先级 / 统一组批的功能验证（117 个测试，含可控时钟与模拟执行器）。
> **910C 真机联调本轮未执行**（无空闲卡），脚本见
> `examples/sched_engine_910c_vllm.py`，执行口径见文末。

## 1. 实际版本与接入层级（事实记录）

| 项 | 实际值 |
|----|--------|
| vLLM | 0.20.2（镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`） |
| vllm-ascend | v0.20.2rc1-a3 |
| 模型 | Qwen3-Embedding-0.6B（本地权重） |
| 芯片 | 昇腾 910C 单卡（容器内 davinci0，相对索引 0） |
| Python（venv） | 3.11（容器 `.venv`，`--system-site-packages` 复用镜像 CANN/torch） |
| 测试 | pytest 8.3.2（容器内运行） |

**接入层级（两层批次，分别表述，不混用）**：

- **外层组批（本引擎控制的）**：`SchedulerEngine` 把请求按兼容键隔离、按
  优先级取队、按长度分桶、按预算组成 `BatchPlan`，每个 `BatchPlan` 对应
  框架适配层的**一次** `LLM.embed` 调用；
- **内部执行批次（本引擎不控制也不宣称）**：vLLM V1 引擎内部自行调度
  （连续组批 continuous batching / 抢占），实际下发到设备的执行批次由其
  内部策略决定。外层批次大小是内部批次的输入上界之一，不是设备批次本身。

## 2. 四层架构

```
┌─────────────────────────────────────────────────────────────────┐
│ L4 设备能力画像 sched_engine.devices                              │
│    ascend-910c(verified) | kunlun/cambricon/thead/biren/enflame  │
│    (pending) —— 批量/Shape/资源预算钳制：min(策略, 设备)          │
├─────────────────────────────────────────────────────────────────┤
│ L3 推理框架适配器 sched_engine.frameworks                          │
│    FrameworkAdapter: submit_batch / get_results / map_error      │
│    vllm-embed(verified，复用 VllmEmbedExecutor) | 其他仅契约      │
├─────────────────────────────────────────────────────────────────┤
│ L2 模型画像 sched_engine.models                                   │
│    ModelProfile: family / cost_metric / feature_schema / status  │
│    15 模型注册表：1 verified + 14 declared                        │
├─────────────────────────────────────────────────────────────────┤
│ L1 调度核心 sched_engine.core（不依赖 vLLM/模型类型/芯片 SDK）      │
│    RequestQueue(兼容键 lane+优先级取队+状态查询)                   │
│    → BatchCoordinator(复用：分桶/预算/封口，扩展 priority)         │
│    → SchedulerEngine(编排+路径开关) → ResultStore(恰好一次回填)    │
└─────────────────────────────────────────────────────────────────┘
```

分层解耦规则：

- L1 只认 `SchedRequest`（payload 不透明）与整数成本，不 import vLLM /
  torch / 任何芯片 SDK；numpy 仅在执行器层出现；
- L2/L3/L4 均为数据契约 + 注册表，状态字段诚实标注（verified/declared、
  verified/pending），引擎在入口拒绝非 verified 的模型与设备——
  **有接口或配置项不等于已支持**；
- 组批决策不重写：L1 的 Planner 就是既有 `dynamic_batching.BatchCoordinator`
  （53 个既有测试全部保持通过），本轮仅做向后兼容的优先级扩展
  （`BatchRequest.priority`，缺省 0 时行为与历史版本一致）。

## 3. 优先级与硬约束语义（确定、可测试）

取队全序（同一兼容键 lane 内）：

```
(-priority, arrival_mono_ns, 提交序号)
```

- 同档优先级：严格按到达时间、再按提交顺序 FIFO；
- 优先级只影响选择顺序，**不得绕过**以下硬约束：
  1. 兼容键隔离（模型ID|模型版本|执行profile 不同绝不混批）；
  2. `max_batch_size`（批次数量上限）；
  3. `max_total_tokens`（批内成本总预算，文本=token、序列=序列长、稀疏=NNZ/样本数）；
  4. `max_wait_ms`（最长等待）：wait_limit 兜底按组内**全部**排队请求的最早
     到达时间计算——高优先级晚到者不能延长低优先级老请求的等待（反饥饿），
     `next_wakeup_mono_ns` 同口径。

离线重放语义说明：所有请求在入队后即对决策核心可见（与既有
`DynamicBatchEmbedder` 一致），`arrival_mono_ns` 用于 FIFO 与等待窗口计算；
高优先级"未来到达"的请求可被选入较早批次，属文档化行为。

## 4. 公开接口与调用示例

```python
from dynamic_batching import BatchPolicy, FakeEmbedExecutor, VllmEmbedExecutor
from sched_engine import (
    SchedRequest, SchedulerEngine, VllmEmbedFrameworkAdapter,
    PATH_LEGACY, PATH_SCHEDULED,
)
from sched_engine.devices.registry import ASCEND_910C
from sched_engine.models.registry import QWEN3_EMBEDDING_0_6B

policy = BatchPolicy(
    model_id="Qwen3-Embedding-0.6B", policy_version="se-910c-v1",
    bucket_upper_bounds=(128, 256, 512), max_batch_size=8,
    max_total_tokens=4096, max_wait_ms=50.0,
)
engine = SchedulerEngine(
    model_profiles={"Qwen3-Embedding-0.6B": QWEN3_EMBEDDING_0_6B},
    framework_adapter=VllmEmbedFrameworkAdapter(VllmEmbedExecutor(llm)),
    device_profile=ASCEND_910C,        # 策略预算被 910C 画像 min 钳制
    policies=policy,
    path_mode=PATH_SCHEDULED,          # PATH_LEGACY = 原路径直通开关
)
compat = engine.submit(SchedRequest(
    request_id="r001", model_id="Qwen3-Embedding-0.6B",
    payload="待嵌入文本", arrival_mono_ns=time.monotonic_ns(),
    priority=1,                        # 2 档：0 常规 / 1 紧急
))
engine.queue_status()                  # 各兼容键 lane 深度/优先级分布/队头
report = engine.run()                  # 单次会话：取队→分桶→组批→执行→回填
vec = report.results.get("r001")       # 恰好一次；漏/重/错位均显式报错
report.dispatch_records                # 每批组成、封口原因、优先级分布
```

核心组件 API 一览：

| 组件 | 关键接口 |
|------|----------|
| `RequestQueue` | `enqueue(request, compat_key)` / `dequeue(key, max_items)` / `dequeue_all()` / `status()` |
| `SchedulerEngine` | `submit` / `submit_many` / `queue_status` / `run(path_mode=...)` |
| `ResultStore` | `declare(ids)` / `commit(batch_id, ids, outputs)` / `get(id)` / `assert_complete()` |
| `FrameworkAdapter` | `submit_batch(payloads)` / `get_results(handle)` / `map_error(exc)`（可选 `token_count`） |
| `ModelProfile` | `status` / `cost_metric` / `feature_schema` / `extract_cost(payload)` |
| `DeviceProfile` | `status` / `max_batch_size` / `max_total_tokens` / `max_sequence_length` |
| `clamp_policy` | `clamp_policy(policy, device)`：min 钳制，冲突显式报错 |

## 5. 已验证 / 待适配矩阵

### 5.1 模型（15）

| 模型 | 家族 | 成本计量 | 状态 | 已验证组合 |
|------|------|----------|------|-----------|
| Qwen3-Embedding-0.6B | text-embedding | token_length | **已验证**（映射/组批：模拟执行器 117 测试；真机链路脚本待跑） | vllm-embed + ascend-910c |
| BGE | text-embedding | token_length | 待适配 | — |
| Jina-Embedding | text-embedding | token_length | 待适配 | — |
| Qwen3-Reranker | text-rerank | token_length | 待适配 | — |
| BGE-Reranker | text-rerank | token_length | 待适配 | — |
| DIN | behavior-sequence | sequence_length | 待适配 | — |
| DIEN | behavior-sequence | sequence_length | 待适配 | — |
| BST | behavior-sequence | sequence_length | 待适配 | — |
| DeepFM | sparse-rec | nnz | 待适配 | — |
| DCN | sparse-rec | nnz | 待适配 | — |
| ESMM | sparse-rec | sample_count | 待适配 | — |
| Wide & Deep | sparse-rec | nnz | 待适配 | — |
| DLRM | sparse-rec | nnz | 待适配 | — |
| PLE | sparse-rec | sample_count | 待适配 | — |
| MMoE | sparse-rec | sample_count | 待适配 | — |

### 5.2 芯片（6）

| 芯片 | 厂商 | 状态 | 批量/Token/序列上限 | 已验证组合 |
|------|------|------|----------------------|-----------|
| ascend-910c | 华为昇腾 | **已验证**（策略预算口径） | 32 / 32768 / 4096 | Qwen3-Embedding-0.6B@vllm-embed |
| kunlun | 昆仑芯 | 待适配 | —（占位，禁止填值） | — |
| cambricon | 寒武纪 | 待适配 | — | — |
| thead | 平头哥 | 待适配 | — | — |
| biren | 壁仞 | 待适配 | — | — |
| enflame | 燧原 | 待适配 | — | — |

注：910C 数值来自 2026-09-23 NPU bench 验证运行采用的策略
（v1-910c-single-card），是"已验证运行使用的预算上限"，**不是设备极限
探测值**；设备极限需专门画像探针实测后更新。

### 5.3 推理框架

| 框架 | 状态 |
|------|------|
| vllm-embed（LLM.embed 外层组批路径） | 已接入（复用 VllmEmbedExecutor；映射正确性经模拟执行器验证，真机脚本待跑） |
| sglang / tensorrt-llm / mindie | 仅扩展契约（`frameworks/base.py`），未实现、未验证 |

## 6. 下一步接入的最小接口

**接入新模型**（以 BGE 为例）：

1. `ModelProfile` 登记（已是 declared）→ 选定具体 series 与 tokenizer；
2. 框架适配器提供同源 `token_count`（与执行入口同一 tokenizer），
   或 `with_token_counter` 绑定已验证计量器；
3. 提交该模型的 `BatchPolicy`（桶上界按已验证长度范围定）；
4. 在真实框架+芯片跑通 `SchedulerEngine` 全链路（32 请求口径）后
   置 verified 并记录 `verified_on`。

**接入新推理框架**：

1. 实现 `FrameworkAdapter` 三方法（`submit_batch` / `get_results` /
   `map_error`），遵守"返回与提交等长同序"契约；
2. token 计量类框架提供 `token_count` 同源计数器；
3. 错误映射覆盖框架主要异常类型 → `FrameworkExecutionError`；
4. 真机验证后登记 `VERIFIED_FRAMEWORKS`。

**接入新芯片**：

1. 实测（而非照抄参数）该卡的批量/token/序列长度执行预算与 Shape 约束；
2. 填写 `DeviceProfile` 并注明数值来源；能力探针脚本建议放 `probes/`；
3. 至少一个模型+框架组合完成 32 请求链路验证后置 verified。

**序列/稀疏模型成本计量**：`input_token_count` 字段语义是"成本计量
单位数"——文本=token、序列=行为序列长、稀疏=NNZ/样本数；`BatchPolicy`
的桶上界与预算按对应计量单位解释即可，核心算法不变。

## 7. 本轮测试证据（容器内可重复运行）

```
docker exec flagos-batch-yxy-dev-910c bash -c \
  "source /workspace/dev/batch_yxy/.venv/bin/activate && \
   cd /workspace/dev/batch_yxy && python -m pytest tests/ -q"
# 结果：117 passed（53 既有 + 64 新增），0 failed
```

新增测试覆盖：

- 队列：入队/取队/状态查询/兼容键隔离/重复拒绝（`test_sched_queue.py`）；
- 优先级与 3 项约束边界：批量恰好/超 1、token 预算恰好/超 1、等待窗口
  含边界；优先级不绕过兼容键/批量/预算/等待（`test_sched_priority.py`）；
- 结果回填：恰好一次/数量不一致/重复/未申报/漏填（`test_sched_result_store.py`）；
- 端到端：32 请求 × 2 优先级 × 3 到达顺序（正序/逆序/交错），5 环节校验、
  逐 ID 向量一致性（错位即失败）、原路径开关对照、可重复运行
  （`test_sched_engine_e2e.py`）；
- 注册表：15 模型/6 芯片/框架契约的诚实状态与钳制/冲突行为
  （`test_model_registry.py`、`test_device_registry.py`）。

## 8. 910C 真机联调口径（待执行）

前置：宿主 `npu-smi info` 确认 davinci0 空闲；`docker ps` 确认 NPU 容器
数未超 DrvMng 并发上限（≈3）。执行 `examples/sched_engine_910c_vllm.py`。

结果记录规则：

- **调度映射正确性**（本引擎职责）：32 条请求均得到且只得到对应结果、
  批次组成符合预算与优先级规则——以 `assert` + 汇总输出为准；
- **模型数值一致性**（平台职责）：baseline vs scheduled 向量对比仅作参考
  ——vllm-ascend 0.20.2rc1 pooling 存在已知跨调用漂移
  （`probes/qwen3_embed_nondeterminism_probe.py`，
  `benchmarks/results/nondeterminism_default_20260923.json`），逐元素一致
  数与 cosine 分别记录，不与映射结论互相替代。
