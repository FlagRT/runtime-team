# FlagScheduler 0.1.0 验证记录

日期：2026-10-21。以下是本轮实际执行结果，不表示完整生产或真机模型验收。

| 检查项 | 实际结果 | 范围 |
| --- | --- | --- |
| 原有测试 | 117 项通过 | 旧动态组批、单次调度、结果存储、设备/模型契约 |
| 全部测试 | **171 passed** | 包含 CLI/配置、常驻 runtime、回环 HTTP、输入边界、失败回填、ID 重排映射、背压、退出排空、并发接入、15 类模型 mock、启动计划保护，以及新增脚本目录/快照/选择测试 |
| 测试运行方式 | CPU 隔离容器，根文件系统只读，无网络/GPU | 仅项目目录可写，已有 pytest/numpy 依赖只读挂载，临时文件位于 `.tmp/` |
| 包构建 | `flagscheduler-0.1.0-py3-none-any.whl` 构建成功 | 无安装到外部环境；产物位于 `artifacts/dist/` |
| 包冒烟 | 从 wheel 直接导入，关闭 site-packages，console entry point 校验通过 | 无第三方依赖的 Mock 三请求完成 |
| CLI 生命周期 | `serve → status → submit → SIGTERM` 通过 | CPU 容器内项目只读挂载，以普通用户运行，完成请求并正常排空退出 |
| 框架发现 | 找到 Conda 多环境 | `vllm` 环境 0.21.0，另有 0.10.0 / 0.29.0+precompiled 的 vLLM 环境、0.5.9 SGLang 及多个 Transformers 环境；不混用版本 |
| Scheduler 只读定位 | 找到 0.21.0 同步/异步 V1 scheduler 与扩展字段 | 记录源码路径与指纹；不读取运行进程的当前 scheduler |
| 同步插件导入 | `FlagScheduler` 校验通过 | 与真实 0.21.0 的 SchedulerInterface 继承/非抽象/构造签名校验，不加载模型 |
| 异步插件导入 | `FlagAsyncScheduler` 校验通过 | 同上；继承原生异步行为 |
| 静态启动计划 | 已生成 `artifacts/vllm-0.21.0-plan.json` | 模型路径仍为占位值，含自定义和原生 argv；未执行启动 |
| Default/Currently | 初始默认双标记、选择项目脚本、回切 Default、按持久选择生成计划均通过 | 在实际 Conda vLLM 0.21.0 环境只读检查，运行结束将本地选择恢复为 Default |
| 原生脚本保留 | 原文件指纹前后一致，所有已发现原生脚本留有只读 SHA-256 快照 | `.state/schedulers/originals/`，不覆盖原生文件或已有快照；同步/异步、环境别名与版本隔离均有测试 |
| NVIDIA 清单 | 4 × RTX A6000，驱动 595.71.05 | 只读设备清单，不代表空闲或完成了推理验证 |
| 文件范围 | Git 改动全部位于 `dev/batch_yxy/` | 未更改仓库根忽略规则、其他子目录或外部 Conda 源码 |

当前没有加载真实模型、替换运行中服务、运行 GPU 推理或测量设备性能。提供的是可执行的节点骨架与真实 vLLM 接口接入能力；示例内部 scheduler 仍继承原生行为。

下一项设备验收需要明确本地权重路径/版本、指定空闲 GPU，并按使用指南执行原路径/统一入口及原生/自定义 scheduler 的数值、请求映射、预算、退出和性能对照。TUI、其他真实后端、SLO 算法及生产级资源隔离依照架构路线后续实现。

## 2026-10-25 GPU3 真机模型运行（vllm-embed × Qwen3-Embedding-0.6B）

宿主直跑（无容器），GPU3（RTX A6000，`CUDA_VISIBLE_DEVICES=3`），Conda `vllm` 0.21.0，本地权重 `/home/ldaphome/yuanxianyi/model/Qwen3-Embedding-0.6B`。入口 `python -B -m flagscheduler run`，`vllm-embed` 后端进程内加载真实模型（runner=pooling，max_model_len=2048，gpu_mem=0.5），36 条混合长度中文请求（2 档优先级，策略 buckets 64–2048 / batch 8 / tokens 8192 / wait 5ms）。证据与配置见 `artifacts/nvidia_gpu3_20261025/`（含 REPORT.md）。

| 检查项 | 实际结果 | 范围 |
| --- | --- | --- |
| 原生 scheduler 路径 | 36/36 completed，全部 1024 维、有限、L2 范数=1.0 | 真实 GPU 推理；同底句 cos≥0.890 / 异底句 cos≤0.602 |
| 自定义 scheduler 路径 | `scheduler select flagscheduler` 后新进程 36/36 completed | 启动参数确认注入 `scheduler_cls=FlagScheduler`，EngineCore 日志确认生效 |
| 两路径数值对照 | max abs diff 2.83e-3，max cos 偏差 1.28e-4 | 自定义 scheduler 继承原生行为，数值等价符合预期 |
| ID 映射 | 两轮各自与输入 36 ID 严格一一对应，无重复/缺失 | 结果按 request_id 回填 |
| 资源回收 | 进程退出后 GPU3 显存恢复 1 MiB | 无残留 |
| 环境恢复 | 测试后 scheduler 选择恢复 default | 快照与目录状态不变 |

限制：两轮单请求时延差异（p50 1115ms vs 112ms）为首轮 warmup/缓存效应，不构成 scheduler 性能结论；未执行原路径 vs 统一入口的吞吐/性能对照与架构文档完整 32 请求验收口径。

## 2026-10-25 GPU3 scheduler 性能对照（A 直连 / B 统一入口 / C 自定义 scheduler）

同机同卡闭环离线回放，192 条混合长度负载，每路径独立进程同 warmup。证据 `artifacts/nvidia_gpu3_perf_20261025/`（含 REPORT.md 与原始输出）。

| 路径 | 吞吐 req/s | p50/p99 ms | 结论 |
| --- | --- | --- | --- |
| A 原路径直连 | 467.6 | 410.5 / 410.5 | 基线 |
| B 统一入口 batch=8 | 411.4（-12%） | 242 / 429 | 小批量调用开销主导 |
| B32 统一入口 batch=32 | 561.3（+20%） | 206 / 315 | 长度分桶+较大预算反超直连 |
| C 自定义 scheduler（同 B 策略） | 417.4（≈B，1.5% 噪声） | 233 / 421 | 插件继承原生行为，无可测开销 |

决策层微基准：BatchCoordinator submit 2.3–2.6 µs/请求、派发 1.9–3.5 µs/请求，占比 <0.3%，非瓶颈。限制：每配置单次运行、闭环回放、未测在线流式到达；<5% 差异按噪声处理。

## 2026-10-26 torch-predict 后端（PyTorch 引擎，DeepFM/DIN/MMoE）

独立仓内首轮功能开发。新增 builtin 后端 `torch-predict`（`flagscheduler/torch_predict.py` + 无 torch 契约层 `torch_payloads.py`）：进程内 PyTorch 引擎，CPU/CUDA 一等路径（device 显式声明不回退，fp16 仅 cuda），批内单次 H2D 搬运 + inference_mode；三个 zoo 契约（DeepFM=nnz/EmbeddingBag 零 padding、DIN=sequence_length/注意力池化、MMoE=sample_count/多头输出），随机初始化固定 seed，`python:` 工厂留真实权重入口。调度零侵入：Planner/协议未改，runtime 仅在 snapshot 增加 device 信息。

| 检查项 | 实际结果 | 范围 |
| --- | --- | --- |
| 全量测试 | 207 passed / 2 skipped | `scripts/test_flagscheduler.sh`，解释器 conda `vllm_add`（Python 3.12.10，torch 2.13.0+cu130），`--basetemp=.tmp/pytest` 仓内；含新增 36 项（payload 契约 18 + 后端/引擎/e2e/调度对照 18） |
| 调度独立性 | mock 与 torch-predict 同策略同请求的批次组成逐字段一致（回归断言） | 执行后端替换不影响调度决策 |
| CPU e2e | 20/20 completed；deepfm avg 10.6ms / din 23.8ms / mmoe 30.2ms；多头输出按请求切片 | 证据 `artifacts/torch_cpu_20261026/`（REPORT.md） |
| 同源计量 | 声明值=特征实际长度，ID 越界接纳即拒；count_cost 不加载模型 | 接纳与执行双重校验，不一致整批失败 |
| GPU 链路 | **待空闲卡**：检查时 4×A6000 均有他人进程（GPU2/3 为 36GB 任务），按红线不动 | 命令见 REPORT.md，空闲后补记 |

备注：随机初始化输出无语义（未训练权重 sigmoid 饱和），证据只覆盖调度/计量/契约链路。catalog 相关测试需 `--basetemp` 指向仓内 `.tmp/`（官方脚本已内置），否则守卫按设计拒绝仓外 state 目录。

## 2026-11-04 v0.2.0：六层调度框架——终态机/取消/超期收割/aging/观测闭环

独立仓第二轮功能开发。落地生命周期层（四终态 completed/failed/timed_out/cancelled + 分段耗时 queue_ms/execution_ms）、排队超期收割（slo_ms 从仅元数据升级为排队阶段强制）、全链路取消（未派发移除即 cancelled；已派发 cancel-pending、执行完丢弃结果，无设备抢占）、aging 防饥饿（policy.aging_slope_ms/aging_max_boost，扫描时动态有效优先级，缺省关闭）；观测层（滑动窗口分位指标 + 结构化事件 + /metrics Prometheus 导出）；决策核演进（反向索引、cancel/sweep_expired、deadline 参与唤醒）不复制算法；新增 planner/execution/observe/profiles 四个子包与 policy-template CLI。配置纯加法扩展（planner_options/event_log/aging），schema_version 保持 1。

| 检查项 | 实际结果 | 范围 |
| --- | --- | --- |
| 全量测试 | 246 passed / 2 skipped | 207 存量全绿 + 39 新增（决策核 14、生命周期 8、观测 8、HTTP 2、模板/配置 7）；vllm_add 解释器，--basetemp 仓内 |
| 终态语义 | 每个接纳请求恰好一个终态；timed_out 不触设备（Blocking 后端 entered 未置位断言）；cancel-pending 执行完丢弃结果 | 冻结时钟/极小 slo_ms/Blocking 三种确定性构造 |
| aging | 等待提升有效优先级可反转派发序、封顶生效、缺省与历史行为逐字段一致 | ManualClock 决策核级断言 |
| HTTP/CLI | /v1/cancel 幂等观察（already terminal）、未知 ID 400；/metrics 文本含 TYPE/分位样本；CLI cancel/policy-template 冒烟通过 | serve 全链路冒烟：submit→cancel→metrics→SIGTERM 干净退出 |
| 模板 | 15 契约的 policy-template 输出全部可被 parse_config 直接接受；zoo 未覆盖契约回落 mock 并附说明 | 配置生成入口（500 模型规模化的种子） |

未做（后续演进）：多实例并发（InstanceRegistry 已留缝）、CostVector 多维预算、自适应桶/在线控制器、执行中抢占、降级链。slo_ms 仍不承诺准时完成，仅排队超期显式终态。

## 2026-11-04 torch-predict GPU3 链路补验与 CPU/GPU 对照

GPU3 空闲时补跑挂账项：GPU opt-in 测试 1 passed（cuda+fp16）；300 条混合负载（deepfm 120/din 90/mmoe 90）双路径热身后正式轮对照。**执行段 p50 CPU→GPU：deepfm 4.01→2.17ms（-46%）、din 4.65→1.89ms（-59%）、mmoe 0.96→0.81ms（-15%）**；deepfm 的 GPU e2e 反升 266ms 系个别批 ~258ms 执行尖刺在单批串行模型下放大为排队（queue_ms p50=264ms），构成多实例并发的实测必要性证据。证据 `artifacts/torch_gpu3_20261104/`（REPORT.md + 双路径逐请求 JSONL + summary）。

补充（2026-11-04 下午）：同 GPU 路径组批策略对照。batch=1→8→32→32+padding≤0.3 吞吐 76.5→103.1→109.1→123.4 req/s（最高 **+61%**），e2e p50 110.6→40.8ms（**-63%**）；批不变性：fp32 max 3.9e-6（批组成不改变输出），fp16 max ~6e-2 / mean ~2e-4（半精度代价）。证据 `artifacts/torch_gpu3_20261104/strategy_*`。

## 2026-11-10 昇腾 910C 真机验证与 vLLM 兼容修复

分支 `xianyiyuan/fix-vllm-embed-and-compat`。环境：昇腾 910C 单卡（容器内 `davinci6`，设备名 `Ascend910_9382`），镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1-a3`，vLLM 0.20.2 / vllm-ascend v0.20.2rc1-a3，Python 3.11.15；模型 Qwen3-Embedding-0.6B（本地无副本，从 ModelScope 下载，1.19 GB / 311 tensors），全程只读挂载。

| 检查项 | 实际结果 | 范围 |
| --- | --- | --- |
| 全量测试（CPU） | **251 passed / 4 skipped** | 原 246 + 新增 5（兼容辅助 4、版本一致性 1）；无 CUDA 主机的 `cuda:99` 用例改为 skip，其失败路径由 `test_cuda_unavailable_fails_loudly` 覆盖 |
| vLLM 并发回归 | **2 passed** | `FLAGSCHEDULER_VLLM_MODEL=... pytest tests/test_vllm_embed_concurrency.py`；修复前同场景必抛 `Already borrowed` |
| 动态组批 NPU 功能检查 | CHECK PASS：24 条全返回、ID 一一对应、1024 维；6 批 `[8,3,1,4,4,4]`，跨桶 0 | `examples/npu_functional_check.py`（仅替换模型路径常量） |
| 调度引擎 32 请求 × 3 到达序 | 各 6 批、32 结果、1024 维；封口 `batch_limit 3 / wait_limit 3`；legacy vs scheduled cos min/mean 1.0000（逐元素一致 0/32，平台已知 pooling 漂移） | `examples/sched_engine_910c_vllm.py`，未做任何补丁 |
| FlagScheduler `vllm-embed`（32 请求，2 档优先级） | **32/32 completed**，ID 全匹配、1024 维；queue_ms p50 7041.9（一次性灌入）、execution_ms p50 82.3 | 修复后无需规避补丁 |
| A/B/C 基准（n=64，mixed） | A 98.5 / B 81.9（-17%）/ C 70.3 req/s；C e2e p50 241ms vs A 650ms；正确性 B/C vs A PASS（max_abs_diff 2.5e-3 / 3.5e-3） | `benchmarks/bench_embedding_batching.py --executor vllm` |

修复内容：

1. **计量 tokenizer 并发（真 bug）**：`LLM.embed` 内部使用 `llm.get_tokenizer()` 的 Rust 对象且不可重入；运行时 `count_cost`（事件循环线程）与 `execute`（线程池）共用该对象时必现 `RuntimeError: Already borrowed`（NPU 单批约 6.6s，首两批后即触发；NVIDIA 因批次快未暴露）。改为 `dynamic_batching/vllm_compat.load_metering_tokenizer()` 从同一模型目录独立加载计量实例，`flagscheduler/backends.py` 与 `dynamic_batching/vllm_embed_adapter.py` 均使用之；不加锁、不阻塞事件循环。
2. **vLLM 池化入口版本兼容**：0.20.x 用 `runner="pooling"`，0.21+ 用 `task="embed"`。新增 `pooling_llm_options()` 按 `EngineArgs.__init__` 的参数名探测；`examples/sched_engine_910c_vllm.py` 与 `benchmarks/bench_embedding_batching.py` 改用该入口（原脚本写死 `task="embed"`，0.20.2 直接 `TypeError`）。
3. **910C 脚本数据生成**：原实现把“词数”当 token 数（`"reqNN"` 额外占 token），32 条中 r23 实际 514 token 超出 512 桶被显式拒绝。改为按真实 tokenizer 收缩到目标 token 预算，保留 128/256/512 桶边界覆盖，不再靠放宽桶上限绕过。
4. **版本元数据**：`flagscheduler.__version__` 改为 `importlib.metadata` 读取（源码运行回退 0.2.0），新增测试与 `pyproject.toml` 对齐。

限制：真机结果均为单轮；bench 未做多轮重复与预热对照，吞吐差异按单次观测记录，不据此下性能结论；910C 上外层组批在该配置下吞吐低于直连（与 NVIDIA A6000 的 +20% 相反），原因待后续多轮验证（外层批次切分与设备内部连续组批的交互）。

### 2026-11-10 组批矩阵对照（910C，n=192）

脚本 `benchmarks/bench_matrix_compare.py`（本分支新增）：mixed 负载 192 条，A=一次直连、B=按 batch 档位不分组、C=同档位+长度分桶；每配置预热 1 轮，实测 3 轮取中位数；A 自身跨轮对比作为平台噪声基线。

第一轮（`max_total_tokens=32768`）：

| 配置 | req/s | e2e p50 ms | p95 ms | 批次 | bs p50 | padding | vs A max_abs / cos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | **110.5** | 1737.7 | 1737.7 | 1 次调用 | n/a | n/a | 基线 |
| B32 | 89.9（-19%） | 1522.3 | 2134.4 | 6 | 32 | 0.562 | 3.8e-3 / 0.99980 |
| B64 | 91.4（-17%） | 1309.9 | 1999.2 | 6 | 34 | 0.479 | 3.2e-3 / 0.99980 |
| B128 | 91.9（-17%） | 1296.3 | 1999.5 | 6 | 34 | 0.479 | 3.0e-3 / 0.99980 |
| C32 | 79.7（-28%） | 555.4 | 2408.0 | 12 | 17 | 0.115 | 3.2e-3 / 0.99974 |
| C64 | 82.3（-26%） | 528.2 | 2332.3 | 10 | 17 | 0.085 | 4.1e-3 / 0.99975 |
| C128 | 85.3（-23%） | 530.4 | 2250.2 | 9 | 17 | 0.093 | 4.1e-3 / 0.99975 |

注：该轮 B64/B128 被 `max_total_tokens=32768` 钳到与 B32 相同的 6 批（bs≈34），batch 档位未被真正拉开。

第二轮（`max_total_tokens=131072`，只跑 B/C，2 轮）：

| 配置 | req/s | e2e p50 ms | 批次 | bs p50 | padding |
| --- | --- | --- | --- | --- | --- |
| B32 | 89.7 | 1517.7 | 6 | 32 | 0.562 |
| B128 | **96.1** | 1411.2 | 2 | 64 | 0.568 |
| C32 | 78.4 | 1764.2 | 10 | 16 | 0.126 |
| C128 | 86.2 | 551.7 | 6 | 12 | 0.111 |

结论：

- **910C 上外层切批没有吞吐收益**：受批次数影响，2 次调用 -13%、6 次调用 -17~-19%；分桶再叠加 3~10 个百分点。直连单次调用仍是吞吐最优。
- **延迟方向一致、吞吐方向相反**：分桶把 e2e p50 从 ~1.4-1.7s 降到 ~0.53s（-65~-70%），但与 NVIDIA A6000 的“B32 吞吐 +20%、p50 -50%”相反——A6000 上外层组批能反超直连，910C 上不能。
- **外层并发不可用**：线程池并发提交 `LLM.embed` 在 910C 上挂起（650 线程、CPU 2.6%、AICore 0%），vLLM 同步 LLM 不是线程安全的；要流水线化只能用 AsyncLLMEngine 或服务化部署。
- **数值**：B/C 与 A 的 max abs diff 3.0-4.1e-3、cos 0.9997-0.9998；A 自身跨轮 bytes 级一致（0.0）。即同组成重复调用是确定的，**批组成变化**才引入 ~3e-3 差异；NVIDIA fp32 批不变性为 3.9e-6，两者不可互相套用。
- 限制：单卡、无 prefix caching、eager 模式；第二轮仅 2 轮；绝对吞吐不可与 GPU 横向比较，只看同机相对趋势。

跨平台数据归档与汇总见 `reports/SUMMARY.md`（NVIDIA 数据在 `reports/nvidia/`，910C 数据在 `reports/ascend910c/`）。
