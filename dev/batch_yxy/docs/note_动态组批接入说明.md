# 动态组批与长度感知分桶 · 接入说明（交付）

> 对应任务：《Qwen3-Embedding-0.6B-动态组批与长度感知分桶开发实现文档.md》
> 分支：`xianyiyuan/batch_yxy`；实现窗口：09-24 ~ 09-30（提前完成核心与联调）

## 1. T0 环境核对事实（2026-09-23 实测）

| 项 | 事实 |
|---|---|
| vLLM 版本 | **0.20.2**（`vllm.__version__`），editable 安装 |
| vLLM 源码位置 | 容器内 `/vllm-workspace/vllm`，与宿主 `dev/batch_yxy/vllm/` **同一 inode**（挂载确认，宿主直改即生效） |
| 引擎路径 | **V1**：`LLM.llm_engine = vllm.v1.engine.llm_engine.LLMEngine`（`vllm/entrypoints/llm.py:89,381`）；启动日志 "Initializing a V1 LLM engine (v0.20.2)" |
| Embedding 入口 | 离线 `LLM.embed()`（`vllm/entrypoints/llm.py:1223`）→ `encode()`，**返回顺序与输入顺序一致**（docstring 契约） |
| 分词/长度来源 | `LLM.get_tokenizer().encode(text)`，`add_special_tokens` 默认 True；实测 Qwen3 tokenizer 自动附加 EOS(151643)。`VllmEmbedExecutor.token_count` 与 vLLM 入口同一 tokenizer，按 `max_model_len` 截断。**未用字符数代替 token 数** |
| Qwen3-Embedding-0.6B | 本地权重 `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B`（1.2G，safetensors+tokenizer+1_Pooling）；`max_position_embeddings=32768`，hidden 1024，last-token pooling；vLLM 加载 dtype=bfloat16，pooler `seq_pooling_type='LAST'` |
| 设备 | 昇腾 910C 单卡（davinci0），容器内 `torch_npu.npu.device_count()==1`；NPU 监控须在宿主 `npu-smi` |
| 现有基线 | `dev/framework-adapter/probes/qwen_embedding_baseline.py` 为 transformers eager 诊断路径，与本任务 vLLM 路径无关；vLLM 组批基线由本次 bench A 模式建立 |
| vLLM 内部调度参数 | 引擎默认值（本次实测 max_seq_len=4096，piecewise cudagraph 至 256，prefix caching/chunked prefill 开启）。**未接入内部 scheduler_cls**（自定义 Scheduler 非稳定公共接口，本次不替换） |

## 2. 采用的接入路径与控制边界

采用实现文档第 5 节 **路径 2（上层 Embedding API）**：

- 组批核心 `BatchCoordinator` 与 vLLM 完全解耦（T0–T3 可用假执行器离线验证）；
- 适配层 `VllmEmbedExecutor` 包装 `LLM.embed`；`DynamicBatchEmbedder` 按 `BatchPlan.payload_refs` 逐批调用 `LLM.embed`，校验返回数量后**按 request_id 逐项回填**（不依赖完成顺序）；
- **控制边界**：BatchPlan 只组织"外层请求分组"（每个 plan = 一次 `LLM.embed` 调用）；vLLM V1 内部仍会自行调度/组批，本模块**不宣称控制实际设备批次**；
- **回退开关**：`DynamicBatchEmbedder(executor)`（policy=None）即 baseline 模式，行为等价于直接调用 `LLM.embed(全部文本)` 原路径；bench 的 A 模式即该基线。

## 3. 实际新增/修改文件（全部在 `dev/batch_yxy/` 内）

```
dynamic_batching/            组批核心包（新增）
├── __init__.py              公共 API 导出
├── types.py                 BatchRequest / BatchPolicy / BatchPlan / compatibility_key
├── errors.py                错误类型（重复ID/超长/未知模型/已关闭/结果数不匹配等）
├── coordinator.py           BatchCoordinator：submit/poll_ready/next_wakeup_mono_ns/flush/close
├── observability.py         padding_ratio、BatchDispatchRecord、summarize_records
├── executors.py             FakeEmbedExecutor（确定性向量+Padding感知成本模型）
├── vllm_embed_adapter.py    VllmEmbedExecutor + DynamicBatchEmbedder（A/B 开关）
└── config.py                policy_from_config / policy_from_json_file
tests/                       53 个用例（见 §6）
examples/                    example_usage.py、npu_functional_check.py、policy_config_example.json
benchmarks/bench_embedding_batching.py   A/B/C 对照脚本（fake/vllm 双执行器）
probes/qwen3_embed_nondeterminism_probe.py  平台缺陷复现探针（证据在 probes/results/）
docs/                        本说明 + 开发实现文档
```

未修改 vllm / vllm-ascend 源码（零侵入）。

## 4. 接口调用方式

```python
from dynamic_batching import (
    BatchPolicy, DynamicBatchEmbedder, VllmEmbedExecutor,
)
from vllm import LLM

llm = LLM(model="/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
executor = VllmEmbedExecutor(llm)          # token_count 与 vLLM 同一 tokenizer

policy = BatchPolicy(
    model_id="Qwen3-Embedding-0.6B", policy_version="v1-910c",
    bucket_upper_bounds=(64, 128, 256, 512, 1024, 2048),
    max_batch_size=32, max_total_tokens=32768, max_wait_ms=50.0,
)
embedder = DynamicBatchEmbedder(executor, policy, model_version="0.6b")
result = embedder.run([("req-1", "文本A"), ("req-2", "文本B")])
vectors = result.vectors                   # {request_id: np.ndarray}，逐项对应无错位
result.dispatch_records                    # 每批可观测记录（§4.3 字段）

DynamicBatchEmbedder(executor)             # baseline 模式 = 原路径回退开关
```

关键语义（详见各 docstring 与测试）：

- 封口原因 `dispatch_reason ∈ {batch_limit, token_limit, wait_limit, flush}`；低流量时 `max_wait_ms` 兜底，不会因批次不满无限等待；
- 兼容键（模型ID|模型版本|执行profile）是硬约束；长度桶是优化分组；相邻桶合并需显式开启且仍受硬预算与 `max_padding_ratio` 约束；
- 超出最大桶上界的请求在 `submit` 时抛 `RequestTooLongError`（明确拒绝，不静默放大桶）；
- `poll_ready` 相同状态重复调用不重复发出（已派发条目即移除）；`next_wakeup_mono_ns` 供调用方安排定时唤醒；
- `close()` 幂等：首次调用 flush 剩余请求（reason=flush）并关闭，之后 submit 抛 `ClosedCoordinatorError`；
- 策略版本固定：`register_policy` 只影响新请求，已排队请求保留提交时快照；
- `padded_length = max(token_lengths)` 为定长 Padding 路径的估算指标；实际设备 Padding 以执行端测量为准。

## 5. 运行方式

```bash
cd /workspace/dev/batch_yxy
.venv/bin/python -m pytest tests -q                    # 全部单元测试
.venv/bin/python examples/example_usage.py             # 无设备示例
.venv/bin/python benchmarks/bench_embedding_batching.py \
    --executor fake --scenario mixed --num 200         # 离线可复现对照
.venv/bin/python benchmarks/bench_embedding_batching.py \
    --executor vllm --model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
    --scenario mixed --num 64 --max-input-tokens 2048  # NPU 真实对照
```

注意：**不要在 `dev/batch_yxy/` 目录下直接 `python -c "import vllm"`**——本目录的 vllm 源码仓目录会以 namespace package 遮蔽 editable 安装（bench 脚本已在导入 `dynamic_batching` 后移除自增路径项规避；如遇 `cannot import name 'LLM' (unknown location)`，换非遮蔽 cwd）。

## 6. 已验证范围

- **单元测试 53 个**（可注入假时钟，无真实睡眠）：桶上界/越界/超模型长度、空输入、重复 ID、单请求/满批/未满批定时/flush、batch 与 token 预算恰好打满与超限、大请求不阻塞、长短混合、相邻桶合并开/关、非相邻桶不合并、Padding 比例约束、低流量桶不饥饿、兼容键隔离（模型/版本/profile）、策略版本切换、结果数量异常显式报错、A/B 向量一致性、配置装载。
- **fake 执行器 A/B/C 对照**（5 场景 × 200 请求，成本模型=固定 10ms + 0.002ms/padded-token，`benchmarks/results/fake-*.json`）：正确性 B/C vs A 逐元素相等（max_abs_diff=0）；吞吐结论按场景如实记录（mixed/long 场景 C 优，short/lowtraffic 场景 A 优——组批等待/多次调用的固有代价，见结果文件）。**fake 成本模型仅验证决策机制，不代表设备性能**。
- **NPU 真实对照**：结果见 `benchmarks/results/vllm-*.json` 与 §7。

## 7. NPU 实测结论（Qwen3-Embedding-0.6B @ 910C 单卡）

### 7.1 重大发现：vllm-ascend 0.20.2rc1 pooling 数值缺陷（组批检测核心输入）

对照实验前的一致性检查发现**平台级缺陷**，复现矩阵（32~64 条混合长度负载，探针
`probes/qwen3_embed_nondeterminism_probe.py`，证据 `probes/results/*.json`）：

| 引擎配置 | 跨调用重复性 | 批内 vs 单条 |
|---|---|---|
| 默认（aclgraph+prefix cache） | 同调用重复 min_cos 低至 **-0.04** | 14/64 偏离 |
| aclgraph，关 prefix cache | 间歇性（一次 -0.037/14 坏，一次干净） | 1/32~14/64 偏离 |
| eager | min_cos 0.65~0.82 | 1/32~5/32 偏离 |
| eager + max_num_seqs=1 | 仍非确定（0.82） | 1/32 偏离 |
| eager + fp32 | **直接崩溃**（CANN Nnopbase executor） | — |
| 任意 + enable_chunked_prefill=False | **直接崩溃**（vllm-ascend `select_common_block_size: No common block size for 16`） | — |

特征：同一 `embed()` 调用内完全自洽（同文本 4 连发 cos=1.0）；**跨调用**相同文本向量漂移、
间歇出现；**返回顺序=输入顺序契约成立**（反序调用逐位对上，cos≈0.9998）。
结论：缺陷与组批方式无关（单请求串行也复现），属 vllm-ascend/CANN 内核或状态问题；
已具备最小复现与证据留存，建议作为独立缺陷上报（对应 0910 拆分表第 62 行组批检测）。

### 7.2 功能联调（T4，CHECK PASS）

`examples/npu_functional_check.py`（eager+关 prefix cache，24 条混合长度）：
24/24 请求全部返回、维度 1024、**ID 一一对应无错位**、批量/预算受控
（6 批，批量 [8,1,4,4,4,3]，封口原因 {batch_limit, wait_limit}）。
baseline vs dynamic 余弦（信息性，受 7.1 缺陷影响**不作验收**）：min=0.69, mean=0.987。

### 7.3 A/B/C 指示性结果（受 7.1 缺陷限制，非验收口径）

`benchmarks/results/vllm-mixed-n64-s20260923.json`（eager+关 prefix cache，
64 条 mixed，max_wait=50ms，bs=32，token 预算 32768）：

| 模式 | wall_ms | req/s | e2e_p50 | 批次数 | 平均批量 | 平均Padding比 |
|---|---|---|---|---|---|---|
| A 原路径（单次全量） | 697.9 | 91.7 | 697.8ms | 1 | 64 | — |
| B 动态组批（单桶） | 818.8 | 78.2 | 818.8ms | 2 | 32.0 | 0.529 |
| C 动态组批+分桶 | 955.9 | 67.0 | 827.7ms | 5 | 12.8 | 0.133 |

如实解读：**该闭环负载下外层组批无吞吐收益**（B/C 各含 ~66ms 等待窗口 + 多次调用固定开销）。
原因：A 模式一次性提交全部输入时，vLLM V1 内部连续组批已高效打包——本模块的外层分组
面向在线流式到达的请求组织（接口/决策正确性已由单元测试与 fake 对照验证），对"整包直通"
场景不构成优化，此结论与实现文档"区分外层请求分组与实际设备执行批次"的预期一致。
数值一致性 verdict=FAIL 由 7.1 平台缺陷导致，与本模块映射无关（顺序契约已单独验证）。

### 7.4 fake 执行器 A/B/C（确定性、可复现，机制证据）

成本模型 fixed 10ms + 0.002ms/padded-token（`benchmarks/results/fake-*.json`），
正确性 B/C vs A **逐元素相等**（max_abs_diff=0）；吞吐按场景：mixed C≈2×A（405 vs 843ms）、
long C 优（807 vs 847ms）、short A 优（47 vs 109ms，组批等待无收益）、
lowtraffic A 优（等待窗口主导，可调 max_wait_ms）。**fake 结果仅验证决策机制，不代表设备性能。**

## 8. 未验证/待补

- **NPU 数值一致性验收被平台缺陷阻塞**（§7.1）：待 vllm-ascend 修复后重跑
  `benchmarks/bench_embedding_batching.py --executor vllm` 的正确性 verdict；
- 在线流式到达负载（真实 QPS/到达序列）下的 A/B/C——当前仅闭环整包与合成突发；
- 设备显存峰值、调度 CPU 开销未采集（需宿主侧 npu-smi 采样脚本，列为待补设备测试）；
- `scheduler_cls` 内部接入路径（非稳定公共接口，按拆分表后续任务处理）；
- 优先级/取消/超时清理/Fallback/降级：仅预留字段（deadline_mono_ns 等），策略未实现（0910 拆分表后续任务）。
