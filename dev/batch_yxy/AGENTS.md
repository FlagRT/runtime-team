# batch_yxy 工作指南

## 用户明确约束：修改范围

项目暂名 **flagscheduler**。用户明确要求：**不得修改 `batch_yxy/` 文件夹外的任何文件**。所有代码、配置、文档、测试文件、临时文件与测试产物都必须留在本目录内。允许只读检查外部推理框架、Conda 环境与设备信息；禁止修改外部框架源码、环境、宿主配置或其他目录的文件。不能为修复忽略规则、安装 CLI、替换 scheduler、添加部署配置等理由扩大修改范围。CLI 安装与启动步骤先交付本目录内的可审查配置，实际部署只按用户明确要求执行。

当前用户目标是节点级 CLI，后续接 TUI；优先 NVIDIA + 当前 Conda `vllm` 环境。用户已授权构建 scheduler 替换能力，采用指定环境的 `scheduler_cls` 启动配置与版本检查，不直接改写已安装框架源码。原有“仅外层组批”边界适用于旧 API；内部 scheduler 集成应单列并准确标注版本与验证范围。

用户要求保留框架原本默认调度脚本，并能在引擎里选择脚本，明确标注 **Default** 与 **Currently**。原文件只读保留，首次建立脚本目录时在本目录 `.state/schedulers/originals/` 保存按 SHA-256 寻址的只读快照；禁止用自定义脚本覆盖原文件或已有快照。`Currently` 在脚本目录中表示保存给后续启动的选择，不代表已切换运行中服务；运行节点的状态另行显示该 binding 启动时的实际配置。同步/异步模式及不同执行环境分别保存选择。

适用于本目录及其子目录。目标是在 Qwen3-Embedding-0.6B 上实现动态组批、长度感知分桶，并在此基础上维护与模型、推理框架、芯片解耦的调度引擎。先核对实际代码与环境，再修改；遵循“先有真实问题数据，再优化”的原则。

## 阅读入口与事实来源

- `README.md`：任务看板、环境入口、当前交付状态。
- `docs/note_动态组批接入说明.md`：已实现的组批接口、控制边界、基准证据和平台问题。
- `docs/note_调度引擎四层架构说明.md`：后续调度引擎的分层、优先级、画像状态、扩展接口及真机验收口径。
- `docs/Qwen3-Embedding-0.6B-动态组批与长度感知分桶开发实现文档.md`：初始设计与 T0–T5 契约；其中“优先级仅预留”等早期范围说明应结合后续架构文档和代码理解，优先级扩展已经实现。
- 仓库根 `../../README.md`、`../../VERSIONS.md`：共享环境、协作纪律和设备红线。

文档里的版本、测试数量和性能数字都是带条件的历史记录。交付时以本次检查与运行结果为准，明确区分已实现、模拟验证、真机验证和待验证。

## 目录与修改边界

| 路径 | 职责 |
| --- | --- |
| `dynamic_batching/types.py`、`errors.py`、`config.py` | 请求、策略、计划、错误与配置校验 |
| `dynamic_batching/coordinator.py` | 组批决策：兼容键、长度桶、预算、等待、排空与优先级 |
| `dynamic_batching/executors.py`、`vllm_embed_adapter.py` | 假执行器、vLLM 执行适配与原路径开关 |
| `dynamic_batching/observability.py` | 派发记录与 Padding 等统计 |
| `sched_engine/core/` | 请求队列、引擎编排、预算钳制与结果回填 |
| `sched_engine/models/` | 模型画像与十五类任务注册表，保留历史接口兼容 |
| `sched_engine/frameworks/`、`devices/` | 框架协议与适配器、设备画像及注册表 |
| `tests/`、`examples/` | 契约测试、离线示例、NPU 功能与调度联调脚本 |
| `benchmarks/`、`probes/` | A/B/C 基准、平台问题探针及各自 `results/` 证据 |

默认在本子目录内完成任务。旧组批 API 通过上层 `LLM.embed` 零侵入接入：一个 `BatchPlan` 对应一次外层框架调用，vLLM V1 仍自行安排内部执行批次。不要将外层计划描述为实际设备批次。新 FlagScheduler 已按用户需求增加独立的内部 scheduler 选择/替换入口，通过启动配置接入，不改写外部框架源码。

调度核心复用 `BatchCoordinator`，不另建一套组批算法。核心保持 payload 不透明，仅处理整数成本，不引入 vLLM、torch 或芯片 SDK；执行细节留在适配层。沿用现有 Python 类型注解、dataclass、显式错误类型和可注入时钟的写法。

## 必须保持的行为契约

- 兼容键 `模型 ID | 模型版本 | 执行 profile` 是硬隔离；长度桶用于优化选择。不同兼容键绝不混批。
- 文本长度来自执行入口同一 tokenizer，包含特殊 token 与截断语义；不得用字符数替代。非文本画像按声明的序列长、NNZ 或样本数计量。
- 同时满足批量、总成本、单请求长度及设备/profile 约束。超出最大已验证桶显式拒绝，不静默放大范围。设备预算按 `min(策略, 设备)` 钳制，冲突显式报错。
- 默认同桶组批；相邻桶合并必须显式开启，并满足预算与配置的 `max_padding_ratio` 硬约束。放不进当前批次的大请求应留给后续批次，不能阻塞其他可执行请求。
- 等待用单调时钟。`max_wait_ms` 是合批等待上限，不是固定睡眠或端到端 SLA；最早排队请求到期时允许发出小批次。`next_wakeup_mono_ns` 与派发判定采用一致口径。
- `poll_ready` 不重复派发；`flush` 排空；`close` 幂等并拒绝后续提交。保留 `batch_limit`、`token_limit`、`wait_limit`、`flush` 等封口原因。
- `BatchCoordinator.register_policy` 仅影响新请求，已排队请求保留提交时策略快照。不要未经核对就把此保证推广到其他层的同名接口。
- 优先级越大越先选择，同档按 `(arrival_mono_ns, 提交序号)` FIFO；默认 `priority=0` 保持既有行为。优先级不能绕过兼容性、批量、成本预算与最长等待；等待兜底要考虑组内全部请求的最早到达时间。
- 离线重放中，提交后的全部请求对决策核心可见，`arrival_mono_ns` 用于排序和等待判断；不要无意改变为按到达时间逐条放行的语义。
- `request_ids[i]`、`payload_refs[i]`、`token_lengths[i]` 与结果第 `i` 项对应。适配层校验结果等长同序，再按请求 ID 回填；漏填、重复、未知 ID、数量不匹配应显式报错。`ResultStore` 保持恰好一次回填。
- 保留基线及回退入口：`DynamicBatchEmbedder(executor)` 的 `policy=None`，以及调度引擎的 `PATH_LEGACY` / `PATH_SCHEDULED`。`SchedulerEngine` 当前是单次会话，`run` 后不再提交。
- 取消、执行超时清理、Fallback、降级、HTTP 服务等应按明确任务另行实现。`deadline_mono_ns` 字段存在不代表已提供完整 SLA 控制。

## 扩展模型、框架与芯片

架构文档登记 15 个模型、6 类芯片；已标记的组合限 Qwen3-Embedding-0.6B + vllm-embed + ascend-910c，其他模型为 declared、其他芯片为 pending，其他框架仅有扩展契约。状态标签应结合实际验证范围解释，不能据此声称调度引擎真机验收已完成。

- 新模型：明确 tokenizer/成本计量器、画像、版本和策略，验证真实模型与框架、芯片组合后再更新 verified 状态及证据。
- 新框架：实现 `submit_batch`、`get_results`、`map_error`，保证返回等长同序；token 计量应与真实执行同源。
- 新芯片：实测预算、长度与 Shape 限制，记录来源。910C 的 `32 / 32768 / 4096` 是历史运行预算，不是探测出的硬件极限；其他芯片不得照抄。
- 按架构文档的 32 请求链路口径验收后再登记新支持组合，未验证入口保持明确拒绝。

## 开发环境与验证

历史环境为 Python 3.11、vLLM 0.20.2、镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`、910C 单卡；运行前核对当前实际版本。容器名 `flagos-batch-yxy-dev-910c`，仓库挂载为 `/workspace`，项目路径 `/workspace/dev/batch_yxy`。

依赖安装与实验在容器内完成，不修改宿主配置或驱动。初始化入口是 `scripts/setup_env.sh`，venv 使用 `--system-site-packages` 复用镜像的 CANN/torch；`requirements.txt` 当前仅有注释，不代表已列全 pytest、numpy 等运行依赖。

在已初始化的容器内：

```bash
cd /workspace/dev/batch_yxy
source .venv/bin/activate
python -m pytest tests/ -q
python examples/example_usage.py
python examples/sched_engine_demo.py
python benchmarks/bench_embedding_batching.py --executor fake --scenario mixed --num 200 --out-dir artifacts/legacy-bench
```

也可从宿主调用已有容器执行测试：

```bash
docker exec flagos-batch-yxy-dev-910c bash -c \
  'source /workspace/dev/batch_yxy/.venv/bin/activate && cd /workspace/dev/batch_yxy && python -m pytest tests/ -q'
```

按改动选择测试：组批看 `test_dynamic_batching.py`、`test_bucketing.py`、校验/配置/版本及结果映射用例；调度看队列、优先级、结果存储、注册表及 `test_sched_engine_e2e.py`。行为修改覆盖预算恰好/超限、桶边界、低流量等待、兼容键隔离、策略版本和结果映射；等待测试用假时钟，避免真实睡眠。文档修改核对路径、命令和事实即可。

### 当前检出差异（2026-10-21 静态核对）

- 初始检出缺少 `sched_engine/models/`；FlagScheduler 构建时已按原契约和测试恢复画像包，并在 CPU 隔离容器验证既有 117 项测试通过。历史 verified 标签仅为旧 API 兼容，不是新增 NVIDIA 认证。
- 仓库根 `.gitignore` 的 `models/` 规则会匹配嵌套同名目录；已仅在本目录 `.gitignore` 添加 Python 源码包例外，没有修改根规则。不要连同模型权重一并纳入版本控制。
- 架构文档末尾将非确定性证据写成了 `benchmarks/results/nondeterminism_default_20260923.json`，实际文件位于 `probes/results/nondeterminism_default_20260923.json`。
- 根 `.gitignore` 忽略 `AGENTS.md`；用户要求推送本次项目工作后，本目录指南通过精确路径显式纳入版本控制，以保留用户的开发约束。根忽略规则保持不变，后续修改本文件按已跟踪源码处理。

## 真机运行与结果记录

启动容器或占用 NPU 前，在宿主运行 `npu-smi info` 确认目标卡空闲，并用 `docker ps` 核对 DrvMng 并发容器限制（历史约 3 个）。多卡实验遵循根 README 的空闲检查要求。不要绕过启动脚本的设备检查；设备冲突时核对归属，避免影响他人的容器或进程。

Compose 启动依赖 `../compose.base.yml`，`!override` 需要 Compose ≥2.24；无对应插件时参考 `scripts/start_container.sh`。该脚本支持 `BATCH_DAVINCI_ID` 选宿主卡，单卡容器内设备相对索引仍为 0。NPU 监控在宿主执行，单卡容器内 `npu-smi` 可能报 `-9005`。

真机入口：`examples/npu_functional_check.py`、`examples/sched_engine_910c_vllm.py`、基准的 `--executor vllm` 和 `probes/qwen3_embed_nondeterminism_probe.py`。共享权重历史路径为 `/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B`，保持只读。若出现本地 `vllm/` 源码目录遮蔽 editable 安装，换到不遮蔽的工作目录，并参照现有脚本的导入路径处理。

必须如实记录以下限制：

- 历史 vllm-ascend 0.20.2rc1 pooling 存在跨调用向量漂移；组批模块的 ID 映射正确性和模型数值一致性分别验收。不能将该历史缺陷未经复测认定为当前版本结论，也不能据此忽略新的映射错误。
- 2026-10-14 调度引擎的 910C 真机 32 请求联调仍为待执行；2026-09-23 的组批 NPU 功能验证不能代替这项验收。
- A/B/C 分别是原路径、动态组批单桶、动态组批加分桶；保持输入、到达序列、模型/tokenizer、设备、并发与预热条件一致。保留无收益或回归结果，不为预期结论修改统计口径。
- fake 执行器只提供确定性机制证据；`padding_ratio = (N * max(lengths) - sum(lengths)) / max(N * max(lengths), 1)` 是长度差异估算，未经测量不能当作设备实际开销。不可得指标明确标记，不用估算冒充实测。
- 新实验结果使用独立文件或输出目录，保留既有 `benchmarks/results/`、`probes/results/` 证据；同步相关接入说明和 README 看板，写明条件、结果与未验证项。

`.env`、`.venv/`、`.opencode/`、凭据、模型权重与本地运行产物不入仓；尤其不要读取并输出 `.opencode` 的认证内容。

## FlagScheduler 当前开发入口

- 新入口 `python -B -m flagscheduler`，安装元数据为 `pyproject.toml`；CLI/Mock 控制面保持标准库依赖，重型执行框架按需加载。
- 新文档：`docs/flagscheduler_使用指南.md`、`docs/flagscheduler_架构与开发路线.md`、`docs/flagscheduler_适配器开发.md`。
- `flagscheduler/` 保存新 CLI、发现、启动计划、常驻 runtime、回环 HTTP、Backend/Planner 协议及 vLLM 版本插件；`sched_engine/` 保留原单次 API。
- `flagscheduler/catalog.py` 管理脚本原件快照、注册、Default/Currently 标记及持久选择；CLI 的 `scheduler list/select/add` 与后续 TUI 共用此 API。未显式覆盖 `scheduler_cls` 时，新的启动计划与 `vllm-embed` binding 使用保存的选择。
- 当前 Conda `vllm` 环境实测发行版为 0.21.0；其他环境可能版本不同，禁止混用。只读 NVIDIA 清单显示 4 张 RTX A6000，设备清单不表示空闲。真实模型/性能验收单独记录。
- 运行测试时关闭外部自动插件，临时文件与缓存放 `.tmp/`，产物放 `artifacts/`。使用现有依赖，不为测试修改外部 Conda 环境；必要时在只读根文件系统的 CPU 隔离容器中执行，唯一可写挂载为本目录。
