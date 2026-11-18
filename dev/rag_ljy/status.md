# RAG 当前状态

更新日期：2026-11-18（北京时间）。当前配置：Milvus/Elasticsearch请求级选择＋双die OneRec vLLM完整RAG链路，默认Milvus。
容器`rag-ljy-vllm-910c`已应用统一启动入口，8088/8089/8090均已接入。

本文件只记录当前能力、运行规则、有效验证结果与待完成事项，固定保留在`dev/rag_ljy/status.md`。

## 当前运行规则与配置（2026-11-18核对代码及在线 `/health`）

以下配置与在线服务一致；规则实现状态和真实实验验证范围分别说明。

### 1. 工作流、模型与设备

| 服务/阶段 | 当前实现与位置 | 运行关系 |
| --- | --- | --- |
| RAG入口 | 8088，同时接入Milvus和Elasticsearch | 接收请求、显存准入、逐阶段调度、组织结果 |
| 向量化 | Qwen3-Embedding-0.6B，vLLM，8089，物理NPU3/die7 | RAG通过HTTP调用 |
| 检索 | Milvus/Elasticsearch，CPU检索 | 使用既有NFCorpus索引，不是NPU加速数据库检索 |
| 精排 | Qwen3-Reranker-0.6B，Transformers，die7 | 保持本地模型初始化/执行线程 |
| 生成 | OneRec-8B，vLLM，8090，物理NPU3/die6 | 8088通过HTTP传入实际构造的提示词token IDs |

每条请求通过`retrieval_backend: "milvus"`或`"elasticsearch"`选择数据库，两类请求可以在同一个8088服务中并发。
未指定字段时使用默认库（当前Milvus，启动配置`RAG_WORKFLOW_BACKEND`只决定默认值）。
两个数据库客户端常驻，共享一套embedding、精排、生成模型和阶段队列；不会为每个数据库重复加载模型。
选择保存在请求上下文，不修改共享默认库，不自动切库或合并两库结果。
响应头`X-Retrieval-Backend`及生成响应`retrieval_backend`标明实际选择；非法或未启用的后端返回422。
启动时核对两个数据库连接和索引；`/health.retrieval_backends`列出已启用后端，`retrieval_backend`表示默认库。

完整生成请求顺序是向量化→检索→精排→生成；不同请求按各自依赖独立推进。
`generate=true`执行生成，未指定时保留原检索/精排返回格式；不是所有请求都强制生成。
当前统一启动明确选择vLLM生成，不加载旧Transformers OneRec，也不自动回退到旧后端。
各设备进程内都可称`npu:0`，但物理映射分别为die7和die6；**两颗die的容量和预算不能相加使用**。

### 2. 显存高低水位：85%暂停、75%恢复

唯一准入显存来源是`device-context.runtime.memory_stats()`，两颗die分别采样、分别保持压力暂停状态。
使用`used = max(used_mb, total_mb - free_mb)`处理读数舍入差异；比例为`used / total_mb`。

| 条件 | 行为 |
| --- | --- |
| 某颗所需die实测占用 **≥85%** | 进入压力暂停，拒绝依赖该die的新请求 |
| 已暂停，占用仍 **>75%**（包括75%～85%之间） | 保持暂停，不因刚降到85%以下就恢复 |
| 已暂停，占用降至 **≤75%** | 解除该die的压力暂停；仍需通过余量、遥测及预算检查才能接收 |
| 从未触发高水位，占用处于75%～85%之间 | 不因这个区间本身暂停，继续按其他条件判断 |

**软件预留预算不计入85%/75%水位判定**，它只影响可用余量检查。
有生成的RAG请求必须同时通过die7与die6检查；不执行生成的请求只依赖die7。
每次采样也更新暂停状态，不要求刚好有请求到达才能记录高水位。
已接收请求不因后续水位升高被取消，继续按阶段执行。

### 3. 5GiB余量与请求预算

| 设备 | 新请求的入口条件 | 预算归属 |
| --- | --- | --- |
| die7：向量化/精排 | `实测空闲 − 未释放预留 − 新请求估算 ≥5120MiB` | 根据查询token、模型形状/精度、精排批量及执行配置动态估算 |
| die6：vLLM生成 | `实测空闲 ≥5120MiB`，同时通过水位和遥测检查 | **生成请求预留为0**，生成并发与KV池由vLLM管理 |

生成入口不附加每请求显存预留；die7的向量化/精排仍使用动态预算。
`generation.reserved_mib=0`表示“不在RAG入口预留生成预算”，不表示生成不消耗显存。
权重、引擎常驻KV池和已保留的设备内存已体现在实测占用中，不按每条请求重复预留。
vLLM显存目标比例0.55是引擎分配配置，**不是85%高水位、计算利用率或硬显存上限**。

当die7预算不够时，可将精排执行批量逐步减小后重新估算；仍处理全部候选，不通过丢弃候选绕过预算。
本地精排执行前继续按实际形状检查，并保留估算异常/低估后的停止新准入机制；动态估算仍标记未完全校准。
die7预算随已接收请求持有至整个请求完成（包括等待远端生成）；完成后还需有效采样**开始时间晚于完成时间**，
才释放待回收预算，防止重复花费旧读数中的空闲显存。
已验证的空闲缓存抵扣仅用于无其他在途/待释放预算时；缓存释放导致实测占用下降时，抵扣额会相应转回预留，避免重复使用。

### 4. 遥测、拒绝与请求生命周期

- 当前采样间隔配置为 **0.01秒（10ms）**，两颗die各有独立采样进程；实际采样频率还受读取耗时影响。
- 最大可接受采样年龄为 **3秒**。读数过期、缺失、无效或估算器故障时，拒绝依赖它的新请求，不沿用旧值放行。
- 统计单位为MiB；当前底层scope仍是`backend_reported_unverified`，运行配置显式允许该scope，物理die映射另行核对。
- 准入拒绝返回 **HTTP503＋`Retry-After: 1`**，记录原因及相关设备/阶段；不会自动重新放回内部队列。
- 客户端断开不取消已经接收的任务；队列和预算继续持有到工作完成。异常会结束该任务并走预算释放流程。
- 生成后端调用失败返回503并记录，不重试、不伪造回答、不自动切回Transformers。
- 请求数量、显存占用和NPU计算利用率是不同指标；本轮50条实验记录了前两者，未测计算利用率。

### 5. 高／中／低优先级与等待老化

| 项目 | 当前规则 |
| --- | --- |
| 请求字段 | `priority: "high" / "normal" / "low"`，调用方指定 |
| 默认/非法值 | 未指定为`normal`；非法值HTTP422 |
| 传播范围 | 同一请求的原始等级传到向量化、检索、精排、生成四个阶段 |
| 出队选择 | 每次出队重新计算有效等级；高等级优先，同级按进入**该阶段队列**的顺序FIFO |
| 插队范围 | 高等级可以超越队列中较早的低等级任务；不打断正在执行或已派发的工作 |
| 老化间隔 | 默认 **30秒升一级**，由`--priority-aging-seconds`配置，不是实测确定的最优值 |
| 老化结果 | low等待30秒→normal，60秒→high；normal等待30秒→high；最高high |
| 计时起点 | 实际获得队列槽位时开始；等待空槽位的时间不计入本阶段老化 |
| 下一阶段 | 按请求原始等级重新进入新队列，等待时间重新计算，提升不永久写回请求 |

30秒是等级提升间隔，不是响应时限或开始执行保证。高优先级无需先等待30秒才能优先出队。
这是RAG各阶段队列的策略，不是跨阶段的全局顺序，也不保证高优先级请求最终最早返回；
vLLM内部批处理/执行仍由vLLM调度。**所有优先级使用相同的显存检查，高优先级不能绕过准入。**

### 6. 队列容量与并发配置

| 阶段 | 等待槽位 | 工作线程/消费者数 |
| --- | --- | --- |
| 向量化 | 8 | 8 |
| 检索 | 8 | 4 |
| 精排 | 8 | 1 |
| 生成HTTP调用 | 8 | 4 |

`max_inflight=0`表示关闭全局请求数量上限；**不是API最多8条请求**。
队列的8是该阶段等待槽位，不包含已经出队执行的任务。已接收请求遇到满队列会等待空位，不丢弃、不驱逐已有任务；
等待空槽位保留原有背压顺序，高优先级进入队列后才参与优先排序。
生成HTTP线程与本地精排线程分离，远端生成等待不占用精排模型执行线程。
API允许的精排批量上限32，请求默认批量4；批量是一次模型执行处理的样本数，不是RAG请求数量。

### 7. 模型执行与输入输出限制

| 配置 | embedding vLLM | OneRec vLLM |
| --- | --- | --- |
| 精度/执行方式 | BF16、eager | BF16、eager |
| 引擎显存目标比例 | 0.15 | 0.55 |
| 最大模型序列长度 | 8192 | 5120（输入＋输出） |
| 引擎活跃序列配置 | 8 | 4 |
| prefix caching | 关闭 | 关闭 |
| chunked prefill | 关闭 | 开启，批处理token上限4096 |

RAG生成输入上限4096 tokens（含系统提示、查询、来源上下文），服务输出上限1024；请求默认输出上限256。
引擎的4条活跃序列和RAG生成队列4个消费者是两项配置，不是4条请求的全局准入上限。
提示词保留查询，按上下文预算截取来源并返回截断标记；向vLLM传入最终token IDs，避免再次套模板。
生成调用使用`temperature=0`，模板关闭thinking；返回原始模型回答、来源、token用量及结束原因。
达到输出上限会标记`finish_reason=length`，HTTP200不等于回答未截断或事实质量已验收。

### 8. 启动、观察与验证口径

`start_vllm_workflow.py`是已应用的容器入口：顺序就绪检查OneRec8090、embedding8089、RAG8088，
复用已有服务时核对模型/设备；新启动服务由统一进程管理。容器启动后看到`WORKFLOW=ready`可执行测试。
数据库须已启动；该入口不会重建数据库。退出先停止接收并等待API排空，再停止自身启动的模型进程，
超出停机等待期限会终止进程；不是无限等待。Compose的restart仍为no，不表示宿主机开机自动启动。

`/health`提供`scheduling`、`stage_queues`、`device_memory_by_stage`、`memory_admission_by_stage`；
生成响应的`scheduling.stages`保留原始/有效等级、出队时间、等待时长、候选数和超越数量，日志也记录dispatch事件。
生成设备控制器仅做准入条件检查，不建立请求预留，其accepted/rejected字段不是vLLM实际完成数；
请求执行结果以HTTP记录和模型响应为准。

已验证：完整50条真实RAG生成、4并发、部分阶段实际优先排序，以及双die规则的CPU/模拟回归。
未完成：新双dievLLM链路真实85%/75%压力验收、真实等待老化提升验收、所有阶段优先排序覆盖、回答质量评估。
具体数值和证据见两库50条及混合并发结果，不能用低占用成功请求替代高水位验收。

## Elasticsearch固定50条验证：通过（2026-11-18，北京时间16:34～16:37）

用户执行`run_priority_generation_50.sh --run`，结果位于
`benchmarks/priority_generation_50/elasticsearch/20261118T083441.833295Z`。
原固定50个ID与文本逐条核对一致，输入SHA256为
`9b0a898ba4e50ccf0081591ba4a2a1ddccc960c112582692a52e90aca09e7429`。
**50/50 HTTP200且生成后端全部vLLM，0个503、0错误、0自动重试；实际在途峰值4。**
生成预留全部0；入口embedding/精排预算1539～3540MiB。

总请求跨度约161.24秒；请求平均9.278秒、P95 17.580秒。
200条阶段出队记录中，14次出队超越较早的低有效等级任务，累计16个超越关系
（embedding 5、retrieval 1、rerank 10）。生成阶段未观测插队；老化提升未观测。
因此已获得真实NPU完整链路及部分阶段优先排序证据，不能称为所有阶段/老化验收全部通过。

794个双die显存样本：die7峰值21.142%、最低空闲49,482MiB；
die6峰值57.093%、最低空闲26,920MiB。无遥测错误，结束两颗die准入ready、在途0、预算0、估算fault为空。
未达到85%，此次不作为双die高低水位压力验收。生成内容有14条达到256-token输出上限，1条与查询相同，
工作流成功不代表回答质量已通过评估。

[完整结果](experiment_outputs/benchmarks/priority_generation_50/elasticsearch/20261118T083441.833295Z/summary.json) ·
[可读生成回答](experiment_outputs/benchmarks/priority_generation_50/elasticsearch/20261118T083441.833295Z/generated_responses.txt)。

## 双库混合并发验证

同一个8088服务同时运行4条完整RAG请求，2条指定Elasticsearch、2条指定Milvus，4/4成功生成。
实际在途峰值4；每条HTTP响应头和生成响应后端均与请求选择一致，生成预留0，无遥测错误。
[混合并发结果](experiment_outputs/checks/vllm_rag_live/20261118T091000.416689Z/summary.json)。
两库50条入口的只读预检均已通过；按请求选库无需重启或重建容器。

## 验证范围与待完成事项

| 项目 | 当前状态 |
| --- | --- |
| Elasticsearch固定50条完整RAG | 已通过：50/50生成成功、4并发、0个503，详见上节 |
| 优先级调度 | 已实现；向量化、检索、精排取得真实优先出队证据；生成阶段优先出队及真实老化提升尚未取得证据 |
| 双die显存准入 | 已实现；高低水位、余量、过期采样及在途完成有模拟回归覆盖；当前vLLM链路真实85%/75%压力验收待执行 |
| 完整回归 | 308项通过，实际执行于2026-11-18；包含双库并发路由、结果隔离、共享模型及50条后端核验 |
| Milvus＋当前vLLM完整工作流 | 50/50完整生成通过、0个503；同端口两库混合4并发实测通过 |
| 回答质量与输出完整性 | 待评估；本次14条达到输出上限、1条复述查询，HTTP成功不代表质量通过 |
| 动态显存估算校准 | 向量化/精排估算仍标记`analytical_uncalibrated`，需要扩大实际输入与批量验证范围 |
| 通用可组合工作流 | 待开发；当前为固定依赖链路＋可选生成阶段 |

验证证据：[完整回归](experiment_outputs/checks/dual_backend/20261118T090656Z/regression.log)、
[双die部署核验](experiment_outputs/runtime/vllm_workflow/20261118T083014.398829Z/verified_dual_die_health.json)。

## 启动、测试与数据位置

容器停止后启动完整工作流：

```bash
docker start rag-ljy-vllm-910c
docker logs -f rag-ljy-vllm-910c
```

看到`WORKFLOW=ready`后运行原50条查询，按请求选择数据库，无需重启：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_priority_generation_50.sh --backend elasticsearch --run
# 或选择Milvus
bash scripts/checks/run_priority_generation_50.sh --backend milvus --run
```

该脚本使用原始50条查询，默认4并发、输出上限256 tokens，高/中/低优先级分别16/17/17条；
每条只提交一次，503不自动重试。生成覆盖、优先出队和老化观测分别报告。
测试入口核对实际后端为vLLM，以及两颗die的独立遥测与准入配置。

| 内容 | 位置 |
| --- | --- |
| 固定50条查询 | `data/queries/nfcorpus-50.jsonl` |
| 模型宿主机目录 | `/home/jliu171/jliu171_data/models/OneRec-8B` |
| 模型容器目录 | `/mnt/onerec/OneRec-8B`（只读） |
| 实验根目录 | `/home/jliu171/jliu171_data/hetero_data/rag/experiments/`；仓库`experiment_outputs`指向此目录 |
| 50条测试输出 | `experiment_outputs/benchmarks/priority_generation_50/<backend>/<UTC时间>/` |
| 可读回答 | 对应运行目录的`generated_responses.txt` |
| 输入、HTTP结果、显存及汇总 | `queries.jsonl`、`requests.jsonl`、`memory.jsonl`、`summary.json` |
| 运行日志 | `experiment_outputs/runtime/vllm_workflow/<UTC时间>/` |
| 统一启动代码 | `scripts/runtime/start_vllm_workflow.py` |
| Compose配置 | `docker_compose/compose.onerec.yml` |

当前统一入口管理服务生命周期；冷启动压力实验需要适配该入口，不能直接套用独立API重启方式。


## Milvus固定50条验证

实际执行于2026-11-18，结果为50/50完整生成成功、0个503、0错误。
平均响应9.533秒、P95 14.874秒；逐请求复核200条阶段出队记录，原始等级传递无不一致。
23次优先出队各超越1个更早的低有效等级任务：向量化7次、检索2次、精排14次；生成阶段未观测超越。
最长阶段排队约2.758秒，未达到30秒老化间隔，因此本轮没有验证真实老化提升。
19条回答达到256-token输出上限，1条复述查询，生成成功不代表回答质量验收通过。
[结果](experiment_outputs/benchmarks/priority_generation_50/milvus/20261118T090101.236292Z/summary.json) ·
[可读回答](experiment_outputs/benchmarks/priority_generation_50/milvus/20261118T090101.236292Z/generated_responses.txt)。
两库各自50条通过与混合并发验证是不同范围；不据此宣称双die85%压力验收通过。

## 两库完整链路水位验收脚本

已准备 `scripts/checks/run_dual_memory_watermarks.sh`，对Milvus、Elasticsearch依次执行同时、错峰、持续3种请求到达模式。
所有压力来自实际完整RAG请求；按查询长度和精排batch分档增加负载，分别验收die7和die6的实测85%拒绝、75%～85%滞回及不高于75%后恢复，预算拒绝不计作高水位证据。
脚本保留当前85%/75%/5GiB规则、生成零入口预留及运行中的模型，不重启或额外分配占位显存。
脚本相关34项测试及只读预检通过。2026-11-18真实压力运行已结束，总结果 **incomplete**：两库各执行同时、错峰、持续3种模式，共276条请求，57条完整生成成功、219条因 `request_budget` 返回503，无运行错误。
Milvus三模式合计30条成功、108条预算拒绝；Elasticsearch合计27条成功、111条预算拒绝。die7实测峰值73.915%，die6峰值57.601%，六轮均未达到85%，因此尚未验证真实高水位拒绝、滞回和低水位恢复。预算503不能替代高水位503。
运行后服务ready、在途请求0、两颗die预留均为0，遥测有效。
[本轮验收结果](experiment_outputs/checks/dual_memory_watermarks/20261118T103814.960230Z/summary.json)。
OneRec预分配KV池可能使生成die无法达到85%，未达阈值必须报告未完成。

```bash
bash scripts/checks/run_dual_memory_watermarks.sh --run
```

输出：`experiment_outputs/checks/dual_memory_watermarks/<UTC时间>/`，包含两库三模式、两颗die的独立结果、请求与显存时间线、配置、模型信息及代码快照。
参数、判定和命令见[双库显存水位测试说明](docs/dual_memory_watermarks.md)。
