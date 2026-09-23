# Qwen3-Embedding-0.6B 动态组批与长度感知分桶开发实现文档

## 1. 任务目标

面向当前正在适配的 vLLM + Qwen3-Embedding-0.6B 推理路径，实现一个可由外部调用的动态组批与长度分桶模块。模块根据分词后的序列长度、模型及执行兼容信息、批量预算和等待窗口形成批次，输出可追踪的 `BatchPlan`；调用方负责实际模型执行和向量结果返回。

本次交付重点是**接口、决策逻辑、最小 vLLM 适配和验证证据**。不建设独立 HTTP 服务，不扩展为覆盖 15 个模型的通用调度平台。优先级、排队请求取消、执行超时清理、Fallback、服务降级、离线任务和其他模型族调优在 0910 拆分表中有后续独立任务，本次仅预留兼容字段，不实现其策略。

### 完成定义

1. 调用方可提交 Qwen3-Embedding 请求，并获取形成的批次计划；低流量时不会因批次不满而无限等待。
2. 同一批次中的请求满足模型/执行配置兼容要求，且不会突破批量和 token 硬预算。
3. 长度桶确实影响候选选择；长短混合场景的决策与 Padding 估算可解释。
4. 批次成员与原请求 ID 一一对应，vLLM 返回的向量可无错位地映射回请求。
5. 有可复现的功能测试和相同负载下的基线对照记录。性能改善只按实测结论表述。

## 2. 开工前核对

开发容器先检查目标仓库，而不要直接假定本地文档中的代码结构与运行环境相同。把以下事实写入实现说明：

- vLLM 的准确版本、V0/V1 路径、Qwen3-Embedding 运行方式（离线 `embed`、服务端 Embeddings API，或项目自有入口）。
- 请求在哪一层完成分词，是否能够拿到包含特殊 token、截断之后的**实际 input token 数**。
- 当前组批入口、vLLM 执行入口、结果回调位置，以及现有测试和性能采集方式。
- 现有最大序列长度、`max_num_seqs`、`max_num_batched_tokens`、设备/Shape profile 限制及已测得的 Batch size 性能数据。
- 是否存在可用的调度扩展点。vLLM 的 `scheduler_cls` 虽可配置，但其自定义 Scheduler 接口不是稳定公共接口；仅在确认当前版本、所需行为和回归范围后接入内部 Scheduler。

如果现有入口无法取得准确 token 长度，先在入口适配层复用同一 tokenizer/截断设置取得长度，避免为了分桶重复使用另一套分词规则。记录长度来源，不以字符数代替 token 数。

## 3. 模块边界与接口契约

按项目语言和现有风格实现；下列 Python 风格接口说明的是语义，不要求照抄类名。

```python
@dataclass(frozen=True)
class BatchRequest:
    request_id: str
    model_id: str
    model_version: str
    input_token_count: int       # 与实际执行输入一致；包含特殊 token/截断结果
    payload_ref: object          # 输入引用，由执行适配层解释；调度核心不复制大张量
    arrival_mono_ns: int         # 单调时钟；避免系统时间调整影响等待判断
    execution_profile: str | None = None
    deadline_mono_ns: int | None = None  # 可选；本次不承诺完整 SLA/执行超时控制

@dataclass(frozen=True)
class BatchPolicy:
    model_id: str
    policy_version: str
    bucket_upper_bounds: tuple[int, ...]  # 严格递增，覆盖已验证长度范围
    max_batch_size: int
    max_total_tokens: int
    max_wait_ms: float
    allow_adjacent_bucket_merge: bool = False
    max_padding_ratio: float | None = None  # 若使用，应说明软/硬约束语义

@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    policy_version: str
    compatibility_key: str
    bucket_ids: tuple[int, ...]
    request_ids: tuple[str, ...]   # 执行输入顺序
    payload_refs: tuple[object, ...]
    token_lengths: tuple[int, ...]
    total_tokens: int
    padded_length: int | None      # 实际执行需要 Padding 时才填写
    dispatch_reason: str           # batch_limit/token_limit/wait_limit/flush 等
    created_mono_ns: int

class BatchCoordinator:
    def submit(self, request: BatchRequest) -> None: ...
    def poll_ready(self, now_mono_ns: int) -> list[BatchPlan]: ...
    def next_wakeup_mono_ns(self) -> int | None: ...
    def flush(self) -> list[BatchPlan]: ...
```

接口要求：

- `submit` 校验重复 ID、长度、模型、profile 和策略范围；非法输入返回明确错误，不默默截断。
- `poll_ready` 可以在新请求到达、执行资源可用或定时器到期时调用；相同状态下重复调用不得重复发出请求。
- `next_wakeup_mono_ns` 让调用方安排定时唤醒；`max_wait_ms` 是合批等待上限，不是固定睡眠时间。
- `flush` 用于进程关闭或测试排空；批次未满也应发出。必要时增加 `close`，其幂等和后续 `submit` 行为要在文档中写明。
- 请求和计划的策略版本固定；更新策略时新请求使用新版本，已排队请求不在无记录的情况下改变规则。
- 若项目已有统一请求/计划类型，映射到现有类型，避免复制一套平行的公共对象。

### 兼容键与结果映射

兼容键至少区分模型 ID、模型版本、执行 profile、输入/输出任务模式和会影响执行语义的配置。**兼容键是硬约束，长度桶是优化分组**；跨相邻桶只在兼容键相同、profile 可执行且配置允许时发生。

`request_ids[i]` 与 `payload_refs[i]`、`token_lengths[i]` 及执行结果第 `i` 项对应。执行适配器应校验结果数量，然后按 ID 返回；不得依赖"完成顺序等于提交顺序"。如果当前 vLLM 返回对象自带请求 ID，优先用 ID 显式关联。

## 4. 组批决策规则

### 4.1 长度桶

Qwen3-Embedding 使用实际 input token 数作为主维度。桶边界配置来自已采集的长度分布、设备限制和 Batch size 基线；不要把示例数字固化为生产默认值。超出最大已验证长度的请求必须有明确处理结果：拒绝、由现有执行路径处理，或进入经验证的备用 profile；不得静默放入最大桶。

初版采用固定边界。为每个兼容键保留按桶组织的候选请求。默认只在同桶挑选；允许相邻桶合并时，必须仍满足硬预算，并记录跨桶原因和 Padding 变化。避免为了等待某个桶满而饿死低流量桶：最早请求达到等待上限时，应发出可执行的小批次。

### 4.2 预算与封口

初版至少同时满足：

```text
batch_size <= max_batch_size
sum(input_token_count) <= max_total_tokens
request_len <= 当前模型和执行 profile 的允许范围
```

如果项目设备执行路径使用定长 Padding、离散 Shape profile 或另外的显存/形状硬约束，再将其加入可行性检查。不要把 `sum(tokens)` 当作显存占用或实际计算量的完整替代。Padding 估算仅在该执行路径确实产生相应 Padding 时作为性能指标；vLLM 内部实际调度/计算方式应由当前版本的测量结果确认。

组批顺序以最早到达请求为种子，在同兼容键、同桶的有界候选范围内贪心加入满足预算的请求。某个大请求放不进当前批次时，应保留给下一批次，不因它阻塞其他可入批请求。以下情况封口：达到硬预算、最早请求达到 `max_wait_ms`、明确的执行资源触发且继续等待无收益，或显式 `flush`。记录具体原因。

如有可信的执行时长画像和请求 deadline，可增加 `estimated_runtime + response_reserve <= earliest_deadline - now` 检查；没有画像时，不宣称能够保证端到端 SLA。当前任务只需把时延目标/等待窗口纳入组批决策。

### 4.3 可观测数据

每批至少记录：策略版本、兼容键、桶 ID、请求 ID 列表、各请求长度、批量大小、总 token、预算上限、等待时间、封口原因和执行结果映射。能测到时记录实际执行 profile、设备执行时长、端到端时延和显存；不能测到的指标标记为不可得，不用估算值冒充实测。

可计算的 Padding 比例为 `(N × max(lengths) - sum(lengths)) / max(N × max(lengths), 1)`。这是批次长度差异指标；仅在执行端确实按最大长度 Padding 时才代表实际 Padding 开销。

## 5. vLLM 接入方式

先实现与 vLLM 解耦的组批核心，再根据第 2 节调查结果选一种最小接入路径：

1. **项目已有调度扩展点**：在该扩展点调用分桶与预算决策，复用现有 vLLM 请求状态、执行和结果回收。检查 V0/V1、异步调度、KV/cache 管理和 pooling 输出的兼容性，不直接替换未理解的内部逻辑。
2. **仅有上层 Embedding API**：在请求入口调用 `BatchCoordinator`，由调用方按 `BatchPlan` 提交输入并映射向量结果。此路径验证外部接口和请求组织；若 vLLM 内部仍会再次调度，报告中要区分"外层请求分组"与"实际设备执行批次"，不得把前者称作已控制内部调度。

不同时启用两个相互独立的等待/组批策略来解释同一项性能收益。必须保留原执行路径作为 A/B 基线及回退开关。若当前版本没有安全可用的内部接入点，完成独立模块与外层适配即可，并在交付说明中准确注明控制边界。

## 6. 开发任务顺序

| 任务 | 实施内容 | 交付检查 |
|---|---|---|
| T0 环境与基线 | 核对仓库、vLLM 版本、分词和执行入口；整理 Qwen3 的长度分布与当前 Batch size 数据 | 一页接入记录；固定测试输入与原路径结果 |
| T1 接口与配置 | 落实请求、策略、计划及错误类型；校验配置、版本和兼容键 | 外部可导入、可调用；边界输入返回确定结果 |
| T2 基础动态组批 | 实现提交、候选队列、预算、定时封口、排空和一次性发出 | 单请求、未满批、满批、大请求、不重复发出测试 |
| T3 长度分桶 | 加入固定桶、可选相邻桶合并、Padding 估算和决策原因 | 长短混合、桶边界、低流量桶不饥饿测试 |
| T4 vLLM 适配 | 把计划接入现有 Qwen3-Embedding 路径，恢复每请求向量；增加开关 | 输出数量、顺序、数值与原路径一致 |
| T5 对照与交付 | 固定负载分别运行原路径、动态组批、动态组批加分桶 | 测试报告、接口示例、配置示例、回退方法 |

T0–T3 可用假执行器完成，不依赖芯片设备；T4–T5 在现有可运行的 Qwen3-Embedding 环境验证。以 0910 拆分表的 9 月 24–30 日为当前实现窗口：先确保接口和正确性，再完成设备对照；如果设备/真实流量暂不可用，交付可复现的固定输入对照，并列明待补的设备测试。

## 7. 必测用例与验收口径

### 功能测试

- 长度刚好落在桶上界、刚好超过上界、超过模型最大长度；空输入及重复请求 ID。
- 单请求、满批、未满批定时发出、`flush` 排空；计时使用可注入单调时钟，不用真实睡眠制造不稳定测试。
- 长短输入混合、相邻桶合并关闭/开启、兼容键不同的请求绝不合批。
- `max_batch_size`、`max_total_tokens` 恰好达到边界及加入下一请求会超限。
- 大请求与小请求交错到达；大请求不能长期阻塞后续可执行请求。
- 一个批次输出与原请求 ID、输入顺序逐项对应；执行返回数量异常时显式报错。
- 策略版本切换后，已有请求和新请求的版本记录正确；回退开关恢复原路径。

### 对照测试

对同一模型版本、tokenizer、输入集、到达时间序列、并发度、设备和预热条件分别运行：A 原路径；B 动态组批无分桶；C 动态组批加长度分桶。输入至少包含短文本、长文本、长短混合、突发与低流量。记录请求数/秒、向量数/秒、P50/P95/P99 端到端时延、排队时间、批量分布、长度分布、理论及可测的实际 Padding、设备执行耗时、显存峰值和调度 CPU 开销。Shape 切换只在后端确有可观测 profile/shape 切换时统计。

正确性先于性能：每个请求得到且只得到一个向量，维度、归一化/后处理方式与原路径一致；数值比较使用项目已有精度容差。吞吐提升是方案目标，但若特定负载无收益或回归，应保留原路径开关、给出负载范围和原因，不为满足结论改动输入或统计口径。

## 8. 交付物

开发容器完成后提交：实现代码、必要的配置示例、外部调用示例、针对上述边界的测试、Qwen3-Embedding 最小联调记录、A/B/C 对照结果及一份简短接入说明。说明中列出实际修改文件、vLLM 版本、采用的接入路径、接口调用方式、已验证范围和未验证的设备条件。不要声称外层计划就是 vLLM 实际设备批次，除非日志或调度跟踪已经证实二者一致。

## 9. 依据

- `功能拆分表-0910.xlsx`，Sheet1 第 62–71 行：当前 Qwen3-Embedding 组批检测与第 63 行动态组批、长度感知分桶任务，以及后续功能边界。
- `实施方案_运行时+模型适配工具链-在线与离线任务调度-袁先乙.md`：兼容性键、动态 Batch、长度桶、多维预算、BatchPlan、结果映射、配置版本和验收证据。
- vLLM 官方文档：`https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/`、`https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/interface/`、`https://docs.vllm.ai/en/latest/models/pooling_models/`。具体接口以开发容器中的已安装版本为准。
