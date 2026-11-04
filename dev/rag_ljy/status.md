# RAG 当前状态

文档更新：2026-11-04（北京时间）。Milvus真实显存验收于2026-10-30完成；Elasticsearch于2026-11-04 19:35～19:37完成。最近完整回归234项通过（2026-11-04）；实验目录时间使用UTC。

**本文件必须保留在 `dev/rag_ljy/status.md`，作为项目当前状态的固定入口，随开发持续更新。**

**当前结果：Milvus与Elasticsearch均已完成真实请求驱动的85%三模式显存准入验收。两后端各自通过突发、错峰、持续到达的高位拒绝、中间滞回、低位恢复及在途请求完成检查，并通过采样过期检查。验证范围为各自记录配置下的独立冷启动实验，不代表任意负载/缓存状态；尚未实现请求优先级调度。**

目前交付形态是可运行的 **RAG检索与阶段调度原型**：支持在线多请求、独立有界阶段队列、
混合检索、精排、设备内存探测及可选内存感知准入。答案生成尚未实现，尚未达到生产部署标准。
本文根据当前源码及已保存的实测证据整理；以下验证日期是实际运行日期。

## 当前完成情况与后续工作（2026-11-04）

| 项目 | 当前状态 |
|---|---|
| 多请求与阶段队列 | 已实现每请求依赖推进、有界阶段队列及中间结果传递；rerank仍为单worker |
| 显存准入 | 动态逐请求预算；实测≥85%暂停，≤75%恢复；新请求预计剩余≥5GiB；503携带Retry-After，拒绝请求不自动重入队，已接收请求继续完成 |
| Milvus专项 | 突发／错峰／持续到达及采样过期检查通过，三模式峰值均85.627%；正式profile `2048:32` |
| Elasticsearch专项 | 相同验收项全部通过，峰值85.443%／85.442%／85.440%；正式profile `2112:32`，最低采样空闲9,134MiB |
| 优先级调度 | 尚未实现；高／中／低优先级及老化机制属于后续计划，不能把三种到达模式称为三级优先级 |
| 生成与可组合工作流 | 生成模型尚未接入；当前为固定向量化—检索—精排流程，阶段按需组合、通用流程上下文等仍待开发 |

两种数据库的通过范围均为记录配置下的独立冷启动实验；不表示任意输入或缓存布局都必然达到85%，也不表示估算是通用峰值上界。数据库检索在CPU执行，尚未验证卡商加速检索能力。

Elasticsearch已通过配置的复现命令：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_elasticsearch_memory_acceptance.sh --run --profiles 2112:32
```

默认`2048:32`曾未达85%，复现成功实验应显式保留上述参数。测试会在每轮开始前准备独立冷启动环境，在每轮内部完成真实高位拒绝、中间滞回与低位恢复；当前更新仅整理已有证据，没有重跑实验或提交、推送Git。

## Elasticsearch最终验收通过（2026-11-04 19:35～19:37，北京时间）

用户执行`bash scripts/checks/run_elasticsearch_memory_acceptance.sh --run --profiles 2112:32`后，[最终汇总](experiment_outputs/checks/memory_acceptance/elasticsearch/20261104T113704Z/summary.json)为`passed`；已核对原始结果，真实后端为Elasticsearch。

| 到达方式 | 实测峰值 | 高位拒绝／中间滞回／低位恢复／在途完成 |
|---|---:|---|
| 同时突发 | 85.443% | 全部通过 |
| 错峰到达 | 85.442% | 全部通过 |
| 持续到达 | 85.440% | 全部通过 |

三轮均采用真实RAG请求，预热profile `1024:32`、正式profile `2112:32`（档位与请求batch上限，实际执行形状以日志为准）；保持85%/75%实测水位、5GiB余量及动态预算。最低采样空闲9,134MiB，高于5GiB；每轮结束约18.38%。[采样过期检查](experiment_outputs/checks/stale_memory/elasticsearch/20261104T113656Z/result.json)也为passed。各轮准备阶段独立冷启动，实验内通过服务正常回收验证恢复，不以重启作为恢复证据。

原始记录：[突发](experiment_outputs/checks/measured_watermarks/elasticsearch/20261104T113512Z/summary.json)、[错峰](experiment_outputs/checks/measured_watermarks/elasticsearch/20261104T113555Z/summary.json)、[持续](experiment_outputs/checks/measured_watermarks/elasticsearch/20261104T113639Z/summary.json)。首次子句超限失败和第二次84.557%未达阈值的记录保留，以下相关“待重跑/尚未通过”描述为历史进度。本次没有重新运行压力实验、提交或推送Git。

## 历史记录：Elasticsearch第二次运行未达阈值（2026-11-04）

[20261104T112549Z](experiment_outputs/checks/measured_watermarks/elasticsearch/20261104T112549Z/summary.json)已正常执行完整预热及正式负载，没有HTTP500或估算fault，但burst峰值仅84.557%，因此为`threshold_not_reached`，套件返回`incomplete`并停止，错峰/持续和采样过期尚未运行。共5条HTTP：初始化1条、预热2条、正式负载2条（1条200、1条request_budget 503）；最少采样空闲9,690MiB。未观察到memory_pressure拒绝，不构成85%验收通过。

实际预热rerank长度1,435 token，批次32+18及15+15+15+5；正式成功请求仍为2,454 token、30+20批次，初始增量预算24,679MiB。预热设备占用约50.61%，低于Milvus通过试验约51.685%；实际缓存布局不同，沿用同一profile不保证相同峰值。当时建议下一步在独立冷启动下尝试`--profiles 2112:32`，小幅增加真实输入长度，仍由原动态预算和5GiB执行保护约束；当时该profile尚未实测；后续已完成并通过，见顶部最终验收记录。未改动85%/75%阈值、预算系数或使用外部分配。

## 历史记录：Elasticsearch首次失败与修复（2026-11-04）

用户运行[20261104T111437Z](experiment_outputs/checks/measured_watermarks/elasticsearch/20261104T111437Z/summary.json)失败：初始化短请求200，两条1,024档预热请求在BM25检索阶段触发`too many nested clauses; maxClauseCount=1024`，返回500；峰值仅18.364%，正式三模式显存验收未完成。服务并未宕机，记录结束时ready、在途0、预留0，估算fault为空。[API错误日志](experiment_outputs/checks/dynamic_memory_admission/elasticsearch/20261104T111417Z/api.log)保留。

已修复Elasticsearch词法检索：普通请求沿用原multi_match；仅遇到明确的查询子句超限400时，按title/text各自analyzer产生词项，每字段保留前256个去重词项，以最多512个term叶子子句构造dis_max检索，保留title的2倍boost。不提高集群上限；其他400继续报告错误。该回退会改变长查询词法排序/召回，不能称检索语义完全不变。embedding与rerank使用的原始完整查询不截短，NPU压力输入长度未被缩小。

234项回归通过。[64条真实长输入的CPU检索验证](experiment_outputs/checks/elasticsearch_long_query/20261104T112002Z/summary.json)覆盖1,024/2,048/4,096/6,144档，均返回检索结果；没有运行RAG/NPU压力请求。重启前预检查已加入这些长查询的实际词法检索，避免仅核对索引数量后漏掉查询上限；脚本遇异常HTTP会直接打印状态及阶段。用户仍需自行重跑以下同一命令，脚本重启API时加载修复：

```bash
bash scripts/checks/run_elasticsearch_memory_acceptance.sh --run
```

此次修复不等于Elasticsearch显存验收通过，原失败记录不修改。未提交或推送Git。

## 历史记录：Elasticsearch实验代码准备（2026-11-04，验收前）

已准备测试代码，**Elasticsearch真实显存三模式专项尚未通过；首次执行在预热检索阶段失败，修复后待用户重跑**；由用户手动启动。此次只读预检查确认容器healthy、索引`nfcorpus-chunks-v1`有3,633篇文档/6,543个分块，全部有向量，最长578 token，小于部署文档上限640。未切换运行API、未提交NPU压力请求。

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_elasticsearch_memory_acceptance.sh --run
```

该入口在每轮开始前将已排空API明确切换到Elasticsearch并冷启动，依次运行突发、错峰、持续三模式，以及采样过期拒绝/恢复检查；每轮内部不重启。使用与Milvus相同的真实请求预热`1024:32`和压力profile `2048:32`、85%/75%水位及5GiB余量。通过health在实验开始和后续采样核验真实后端，禁止将Milvus结果标为Elasticsearch。Elasticsearch检索候选可能不同，实际padding/显存峰值须重新测量，不能保证照搬profile就达到85%；未达到阈值或缺少滞回证据时返回非零并停止后续轮次，不生成通过汇总。

不带`--run`只做只读索引/长度核验及配置预览。每次重启前再次核对索引、向量映射、数量和长度上界；失败时不停止原API。新API启动失败时尝试恢复原命令及原环境。运行结束保留Elasticsearch后端；若需恢复Milvus，排空后执行`docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c .venv/bin/python scripts/runtime/restart_memory_api.py --run --retrieval-backend milvus`。

新输出按后端归档到外部experiments目录的`checks/{dynamic_memory_admission,measured_watermarks,stale_memory,memory_acceptance}/elasticsearch/<UTC>/`；三模式及采样过期检查均通过后自动核验、汇总。保存索引/语料/tokenizer指纹及原有输入、源码、模型版本、HTTP与实测时间线。原Milvus历史目录不变。

代码验证：完整回归227项通过；随后新增的编排检查与后端检查共8项通过（其中5项已包含在227项中），使用模拟Docker验证只读默认、三模式传参和失败停止；不等同于Elasticsearch真机压力验收。未提交或推送Git。

## Milvus最终验收结果（2026-10-30 00:25，北京时间）

[汇总与审计](experiment_outputs/checks/memory_acceptance/20261029T162525Z/summary.json) · [显存/预算分开展示的时间线](experiment_outputs/checks/memory_acceptance/20261029T162525Z/memory_timelines.svg) · [复现说明](docs/watermarks-experiment.md)

| HTTP到达方式 | 实际到达 | 正式负载HTTP | 实测峰值 | 75%～85%拒绝样本 | 最终实测 | 全部验收项 |
|---|---|---:|---:|---|---:|---|
| 同时突发 | 两条间隔0.629ms | 2（1成功、1预算拒绝） | 85.627% | 83.963%、80.520%、75.947% | 18.39% | passed |
| 错峰到达 | 两条间隔0.211s | 2（1成功、1预算拒绝） | 85.627% | 83.533%、75.947% | 18.39% | passed |
| 持续到达 | 约5QPS开放循环，达到高位后停止增长负载 | 8（1成功、7预算拒绝） | 85.627% | 83.533%、76.517%、75.376% | 18.39% | passed |

- 原始完整通过记录：[突发](experiment_outputs/checks/measured_watermarks/20261029T162300Z/summary.json)、[错峰](experiment_outputs/checks/measured_watermarks/20261029T161940Z/summary.json)、[持续](experiment_outputs/checks/measured_watermarks/20261029T162114Z/summary.json)。每轮同一个采样进程从开始运行至恢复完成，API未在该轮内重启。
- 每轮先1条短请求初始化，再2条约1,024-token真实请求预热；正式压力输入约2,048 embedding token，候选50、请求batch上限32，实际准入计划30，forward按真实形状执行。高位时各轮`inflight=1`；不能把累计请求数或候选batch当作并行RAG请求数。三轮分别共20/20/25条HTTP（包括初始化、预热、负载、拒绝及恢复探测）。高水位压力主要来自真实rerank计算及其缓存，不是vLLM单独的自然增长。
- 三轮采样最低空闲均为9,019MiB（约8.81GiB），高于5GiB；已接收请求均完成，估算fault为空，服务日志未发现OOM异常。采样不是连续时间峰值证明。请求数量、设备实测、软件预留及计算利用率分开；NPU计算利用率未测。
- 高水位以前的预算拒绝也实际验证：实测51.685%、空闲30,317MiB、已预留24,005MiB、新请求估算3,638MiB，剩余仅2,674MiB<5,120MiB，返回503+Retry-After，原因`request_budget`。这不计作高水位拒绝。
- [采样过期真机检查](experiment_outputs/checks/stale_memory/20261029T162334Z/result.json)通过：仅SIGSTOP独立采样进程3.4秒，旧读数导致`stale_telemetry` 503，接受计数不增；SIGCONT后新读数恢复，真实请求200。独立恢复看门狗与finally清理均已配置。
- 最终部署：[配置及日志](experiment_outputs/checks/dynamic_memory_admission/20261029T162235Z/)。真实水位85%/75%、5GiB、动态逐请求估算、`max_inflight=0`；222项回归通过，未提交或推送Git。

### 通过验收时的配置（2026-10-30补充）

本节根据上述三轮保存的部署配置、请求记录和逐forward日志补充；文档更新不代表重新执行了一轮实验。以下是通过验收时的配置，不能把后面的历史版本配置混用。

| 配置项 | 验收值及含义 |
|---|---|
| 设备与总显存 | 宿主机`/dev/davinci1`，容器`npu:0`，device-context报告总量62,748MiB |
| 模型与数据库 | Qwen3-Embedding-0.6B / vLLM；Qwen3-Reranker-0.6B / Transformers；Milvus，6,543个文档块；两模型BF16 |
| 高低水位 | 设备实测占用≥85%暂停；暂停后75%～85%仍暂停，仅≤75%解除压力暂停；软件预留不参与水位判断 |
| 新请求余量 | `实测空闲−未释放预算−新请求增量估算≥5,120MiB`才可接受；解除压力暂停不豁免此项 |
| 请求预算 | `qwen3-execution-v3`动态估算，非固定2GiB；按token、模型/dtype、执行batch和可复用缓存核算 |
| 请求与阶段并发 | `max_inflight=0`关闭入口数量拒绝；embedding worker 8、retrieval worker 4、rerank worker 1；阶段等待槽8不等于最多接受8个HTTP请求 |
| 请求内batch上限 | `--max-reranker-batch-size 32`；每个请求还会规划实际batch，forward前按实际形状检查并可缩小 |
| 长度保护 | 模型最大长度8,192 token；部署文档上限640 token，当前6,543块实测最长578，不截断当前语料 |
| 采样与过期 | 唯一设备指标为`device-context.runtime.memory_stats()`；独立进程采样，实验配置间隔0.001秒，读数最大年龄3秒；实际读取耗时计入年龄，不保证每1ms都有样本 |
| 缓存管理 | `garbage_collection_threshold:0.70`、allocator fraction 0.99；无在途任务且压力暂停时由服务回收闲置缓存；公开C++回收接口释放GIL，使回收期间API仍可处理请求 |

融合attention经首个真实forward算子记录确认后，估算采用1.15执行裕量、每阶段512MiB工作区及单独分配器增长预算；确认前用更保守的dense回退。缓存抵扣只针对已计入实测占用、完全闲置且可复用的缓存段，并独占登记；释放缓存时把相应抵扣补入未完成请求预留。因此“新增预留预算”“forward工作峰值”“设备总占用”是三个不同量，不能直接互相替代。估算仍标记`analytical_uncalibrated`，不声称是任意输入的已证明上界。

### 实验输入、实际batch与显存增长过程

三轮使用同样的初始化和预热，再改变**正式HTTP请求的到达方式**。每条预热/正式负载请求均检索top100、精排50个候选、返回5个结果。这里的rerank batch表示**同一条查询一次forward处理多少个候选文档**，不是同时执行多少条HTTP查询；embedding每条请求的输入batch为1。

| 阶段 | HTTP请求及输入档位 | 实际embedding输入长度 | 准入前rerank长度上界 | 实际rerank padded长度 | 请求batch上限 → 计划batch → 实际forward批次 |
|---|---|---:|---:|---:|---|
| 初始化 | 1条短查询，用于模型初始化及融合算子确认 | 28 token | 800 token | 509 token | 1 → 1 → 逐候选执行 |
| 预热第1条 | `1024`档，4,343字符 | 1,044 token | 1,816 token | 1,431 token | 32 → 32 → **32 + 18**，合计50候选 |
| 预热第2条 | 与第1条相同文本，同时提交 | 1,044 token | 1,816 token | 1,431 token | 32 → 15 → **15 + 15 + 15 + 5**，合计50候选 |
| 正式负载中被接受的请求 | `2048`档，9,109字符 | 2,068 token | 2,840 token | 2,454 token | 32 → 30 → **30 + 20**，合计50候选 |

这些实际长度和批次在三轮成功记录中一致。`1024/2048`是构造输入的档位标签，不是最终模型输入长度；embedding格式化文本后的token数，以及rerank拼接查询、候选文档、提示模板并padding后的长度分别记录。准入时尚未检索到候选文档，因此先使用受文档长度限制约束的上界；检索后才得到表中的实际forward长度。

预热两条请求同时到达，但第2条因第1条已占用预算，计划batch被降到15；全部50个候选仍被评分。正式负载请求虽然指定batch32，实际计划为30，余下20个候选在第二个forward完成。预热两条的初始预算分别为30,613MiB和15,125MiB；正式被接受请求的增量预算为24,005MiB，含已核验缓存复用抵扣。这些数值是本轮计划的结果，不是对所有请求固定使用的预算。

显存增长的具体原因与过程：

1. **独立冷启动基线约17.76%。** 每轮开始前重启已排空API，只为统一初始条件；其后的初始化、预热、升高、拒绝、回落和恢复均在同一次API运行内完成。
2. **真实预热请求使占用升至约51.685%。** 1,431-token的query/document序列以最多32候选一批执行，产生激活、KV、输出张量等实际模型内存；forward结束后，部分内存由allocator缓存以供复用，因此设备实测占用不会立即回到空载值。
3. **更长的正式请求把占用推到85.627%。** 实际序列从1,431增至2,454 token，首批仍处理30个候选。当前reranker前向产生完整序列的词表logits（`logits_to_keep=0`），之后才取末位置用于评分；这部分张量规模与`batch × padded长度 × 词表大小 × dtype字节数`有关。更长序列和较大候选batch会增加实际张量内存，既有缓存布局也影响新增分配，不能按请求数量做线性外推。
4. **峰值时只有1条RAG请求在途。** 85.627%包含模型常驻内存、前面真实请求留下的缓存和当前真实forward的内存；不是多条RAG都在NPU并行计算，更不是额外程序填充显存。vLLM的既有池配置未扩大；本轮增量压力主要来自rerank。
5. **在途请求完成后服务正常回收缓存。** 每轮都采到了75%～85%区间的`memory_pressure` 503，再在≤75%后验证真实请求200，最终约18.39%。客户端没有调用清缓存接口，实验内没有重启制造恢复。

正式负载分别为：突发2条同步释放（实测间隔0.629ms）、错峰2条间隔配置0.2秒（实测0.211秒）、持续按0.2秒节拍开放循环提交（5QPS，配置最长5秒，检测到高位后提前停止增长负载，本轮共8条）。三种模式各接受1条正式负载，其余分别1/1/7条在高水位前因预算不足拒绝；高位与回落阶段另有真实探测请求验证503/200，不能混入正式负载计数。所有初始化、预热、负载和探测合计分别20/20/25条HTTP。

证据可交叉核对：[突发逐请求与采样记录](experiment_outputs/checks/measured_watermarks/20261029T162300Z/events.jsonl)、[突发逐forward及估算日志](experiment_outputs/checks/dynamic_memory_admission/20261029T162235Z/api.log)、[部署参数和环境](experiment_outputs/checks/dynamic_memory_admission/20261029T162235Z/config.json)。另外两轮的原始目录已列在本节上方。

### 复现上述配置和实验

在当前宿主机仓库目录运行；不带`--run`只预览，带`--run`会在每个独立实验开始前重启已排空API，随后依次执行三模式与采样过期检查：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_measured_acceptance.sh --run
```

每轮实际压力命令使用以下参数，`MODE`依次为`burst`、`staggered`、`continuous`；完整脚本还负责每轮开始前的环境准备，不能只在暖缓存状态下连跑下面三次并假定结果相同：

```bash
docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python -u scripts/checks/check_measured_watermarks.py --run \
  --modes "$MODE" --count 2 --warmup-profile 1024:32 --profiles 2048:32
```

全部实验输出保存到`/home/jliu171/jliu171_data/hetero_data/rag/experiments/`，仓库`experiment_outputs`为其符号链接。更详细的启动条件、桥接构建及输出说明见[实验操作文档](docs/watermarks-experiment.md)。

### 验收边界与保留的失败记录

三轮采用相同的独立冷启动准备；开始前的重启仅建立初始条件，不计作任何恢复。每轮内部的高位→中间→低位均由正常请求执行和服务缓存回收产生。直接在前一轮回收后的暖分配器上复用同一profile，曾只达到84.18%～84.20%，相关整体实验仍保留failed/incomplete，不能删掉或改成passed：[16:09](experiment_outputs/checks/measured_watermarks/20261029T160912Z/summary.json)、[16:13](experiment_outputs/checks/measured_watermarks/20261029T161327Z/summary.json)、[16:16](experiment_outputs/checks/measured_watermarks/20261029T161635Z/summary.json)。这说明固定请求数不决定显存峰值，不能声称同一命令在任意缓存布局下必定达到85%。

估算为有实测校核的解析模型，仍非所有输入/config的通用上界；旧97.17%超保护线、外部分配实验及450次请求仅44.629%的历史结果继续保留，不是本次通过依据。

## 历史真机进展（2026-10-30 00:04，北京时间）

已由真实RAG请求安全达到85.515%：[pilot摘要](experiment_outputs/checks/measured_watermarks/20261029T155405Z/summary.json)。观察到10次memory_pressure拒绝、已接收请求跨高水位完成及服务恢复，最少空闲9,089MiB；尚缺中间75%～85%的拒绝证据，因此该轮仍为incomplete，不能称三模式通过。

- 当前执行估算为v3：对完全闲置的真实缓存段给予独占增量抵扣；device-context读数下降时将已回收抵扣额补入未完成请求预算，防止并发重复花费。缓存释放前立即使旧缓存快照失效。
- 新公式15%执行裕量与512MiB阶段workspace，另有分配器增长项。[2,707条历史forward复核](experiment_outputs/checks/estimate_calibration/20261029T154517Z/retrospective.json)无低估，最大已观测峰值/新估算0.962；这是已测范围的回顾性验证，不是任意形状的证明。
- [实时数据库长度核对](experiment_outputs/checks/corpus_token_bounds/20261029T154916Z/summary.json)：当前Milvus的6,543块最长578token，部署`RAG_RERANK_DOCUMENT_TOKENS=640`，当前语料不截断。未来更长文档受此显式配置约束。估算器不再使用无依据的1,024-token文档上界。
- allocator GC配置为0.70、进程ceiling0.99，均不预分配显存。闲置压力回收属于服务正常资源管理；测试客户端不清缓存、不用重启充当恢复。公开NPU Python回收绑定持有GIL，因此新增一个调用同一公开C++回收API的CDLL桥接，释放GIL使采样/HTTP可在回收期间继续；不添加睡眠、不分阶段操纵显存。桥接源码、构建命令、运行时/ABI/二进制哈希保存在`experiment_outputs/runtime_extensions/82f0cce02d30e7b44704/`。
- 220项回归通过。[最新部署](experiment_outputs/checks/dynamic_memory_admission/20261029T160353Z/)将采样改为1ms，正在重验完整滞回，再进行三模式最终验收；以下较早配置为历史记录。

## 历史估算与真实压力修正（2026-10-29 23:24，北京时间）

用户要求继续直至真实85%三模式目标，而非停在预算拒绝。当前部署升级为`qwen3-execution-v2`，见[部署记录](experiment_outputs/checks/dynamic_memory_admission/20261029T152124Z/)。85%/75%实测判定及5GiB余量保持不变，下面较早的shape-v1配置不是当前实现。

- 通过真实请求CPU算子剖析确认`aclnnFlashAttentionScore`、`npu::npu_fusion_attention_v3`，并核对本地vLLM的预分配KV池；未确认融合算子前仍用dense保守回退。新估算按融合attention及层临时量/输出层的不同生命周期计算，保留25%工作峰值裕量、阶段workspace及单独分配器增长预算，不再把不存在的多头完整注意力矩阵计算进去。
- 同一请求的全部候选在CPU统一tokenize、统一padding长度；候选文档仍全部评分，不减少候选来降低显存。先前765次forward记录的新公式对比最大已观测峰值/估算约0.738；独立8请求检查通过，仍不称任意负载下已证明的上界。
- [独立真机检查](experiment_outputs/checks/dynamic_estimator_live/20261029T151058Z/summary.json)覆盖8条成功请求，最末显存28.24%；大batch32请求在该版仍预算拒绝。随后新增正式准入前的实际batch规划：以请求batch为上限选择能满足5GiB的批次，预算对应真实执行计划，响应头`X-Rerank-Planned-Batch`公开计划；即使batch1也不够时仍503，不自动重排拒绝请求。
- 新增正常服务缓存回收：native allocator GC阈值0.65、进程allocator ceiling0.99（均不分配显存），实测压力暂停且无在途任务时回收闲置缓存。测试客户端不清缓存、不重启制造恢复。采样改为5ms以捕获短暂过渡；503携带实际决策采样，证据不依赖HTTP返回后水位仍未变化。
- 215项回归通过。较小批次的扩展循环到49.56%后被主动停止并排空（[记录](experiment_outputs/checks/measured_watermarks/20261029T151238Z/summary.json)保留中断状态）；正在执行针对实际大批次的独立pilot，不把此阶段当三模式通过。

## 最新准入规则（2026-10-29 22:42，北京时间）

已按用户确认部署：设备实测占用≥85%暂停、≤75%恢复，唯一设备读数来源为`device-context.runtime.memory_stats()`；软件预留预算不参与水位触发。未暂停时仅在`实测空闲−未释放预算−新请求动态估算≥5120MiB`时接受。每次采样即更新滞回，避免无请求到达时漏记越线。未知/过期读数拒绝，已接收工作排空后等待新采样释放预算。health新增`watermark_basis=device_measured_usage`；`effective_used_percent`仅作诊断。

[部署记录](experiment_outputs/checks/dynamic_memory_admission/20261029T144130Z/)；212项回归通过，涵盖精确85%/75%边界、预算不触发水位、精确5GiB余量及无HTTP流量时采样锁定压力。新[真实三模式入口](scripts/checks/check_measured_watermarks.py)已完成一轮有限配置测试，三个模式均未达到85%；旧线性加压脚本及安全暂停历史保留。下方旧1GiB及“实测+预算触发水位”的描述仅代表历史版本。

### 新规则三模式真机结果（2026-10-29 22:42～22:44，北京时间）

[原始摘要](experiment_outputs/checks/measured_watermarks/20261029T144248Z/summary.json)、同目录events/输入/源代码快照/模型哈希已保存。全部请求为真实`POST /query`，没有额外显存分配、改小预算、手工清缓存或实验期间重启。

| 模式 | HTTP请求数 | 成功200 | 预算不足503 | 实测峰值 | 验收状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| 同时突发 | 108 | 11 | 97 | 44.629% | threshold_not_reached |
| 错峰到达 | 108 | 10 | 98 | 44.629% | threshold_not_reached |
| 持续到达 | 234 | 17 | 217 | 44.629% | threshold_not_reached |

- 共450次调用，38次实际完成模型链路，412次均为request_budget拒绝，无传输/服务错误，未出现memory_pressure拒绝；不能声称完成85%拒绝/75%～85%滞回/低水位恢复真机验收。总状态incomplete，退出码2。
- 观测到诊断用“实测+预算”最高94.414%，但设备实测未到85%且准入未因该诊断值暂停。360条采样显示预算合计超过85%时仍ready，符合新规则；新请求仍须单独满足5GiB余量。
- 最少实测空闲34,744MiB，未触碰5GiB保护线。结束时服务ready、inflight=0、reserved=0、估算fault为空；该正常空闲状态不是高压后恢复证据。
- 当前大请求估算先受余量条件限制：结束时空闲34,744MiB，扣除5,120MiB后可用预算29,624MiB；6,144-token/batch1请求估算39,408MiB，因此正确返回request_budget。当前保守解析估算及负载形状尚未使设备达到85%，仍需校准估算/负载，不能靠增加被拒绝请求数补齐验收。
- 本轮持续到达按0.2秒开放循环，每档26次、发送窗口约5秒（末尾等待使每档约5.2秒）。已修正后续脚本的计数边界为每档25次；本轮450次原始证据不改写。
- 未提交或推送Git。

## 当前部署（2026-10-29）

磁盘重建后已重新上线：两个模型挂载自 `/home/jliu171/jliu171_data/hetero_data/models`；
Milvus（内嵌etcd，单容器）和 Elasticsearch 数据仍写入本地 RAID 挂载目录。
双库各重建6,543块，BM25/dense冒烟通过；RAG API `127.0.0.1:8088` 当前使用Milvus，
vLLM embedding `127.0.0.1:8089`，宿主机 `/dev/davinci1` 映射为逻辑 `npu:0`。
两轮12次请求全部HTTP成功；20ms错峰校验通过，2秒错峰成功但未重叠，因此该轮校验器的重叠条件未通过。
真实内存探测与准入已开启，本轮未验证压力拒绝/恢复或校准预算。
服务进程在容器内运行，容器重启后须重新启动模型/API服务。
[上线步骤、配置和完整证据](artifacts/rebuilds/recovery-2026-10-29/report.md)。

## 后续实验输出位置

自2026-10-29起，新实验默认写入宿主机
`/home/jliu171/jliu171_data/hetero_data/rag/experiments/`，按 `benchmarks/`、`checks/`、
`diagnostics/`、`logs/`、`rebuilds/` 分目录保存。
仓库 `experiment_outputs` 是指向该目录的本地符号链接；Compose通过 `.env` 的 `RAG_OUTPUT_DIR`
将同一路径挂入容器，已实测容器写入后宿主机可读。
现有检查、诊断和基准脚本的默认输出已改用该链接，历史 `artifacts` 和基准输入保持原位。
显式指定 `--output-dir` 时以该参数为准；请使用 `experiment_outputs` 下的目录。
挂载重启后再次完成串行／同时／20ms错峰共6请求验证，全部通过；结果已实际写入
`experiment_outputs/checks/multi_request/2026-10-29_160106-milvus-757755/`。
该链接与 `.env` 为本机配置，不纳入Git；新机器需创建自己的目录/链接并配置挂载。

## 已实现功能

| 模块 | 当前实现 |
|---|---|
| 数据处理与入库 | NFCorpus原始3,633篇文档按1200字符、重叠150字符切为6,543块，保留文档与块ID；同一批文档向量复用于两个数据库。 |
| 文本向量化 | 文档和查询默认使用Qwen3-Embedding-0.6B的vLLM服务，输出1024维向量；CLI、API、入库与基准使用同一embedding工厂。 |
| 混合检索 | 通过统一数据库接口选择Milvus或Elasticsearch，在所选数据库内执行BM25和dense检索，支持顺序／并发两种模式，再进行RRF融合。 |
| 结果精排 | Qwen3-Reranker-0.6B通过Transformers执行，返回排序片段、来源及分数；当前精排后端仍是Transformers。 |
| 在线多请求 | FastAPI接受同时及错峰到达的HTTP请求；每个请求拥有独立asyncio任务、请求ID及中间结果，按自身依赖推进。 |
| 独立阶段队列 | 向量化、检索和精排各使用一个有界`asyncio.Queue`；每个任务通过自己的Future接收结果，完成后即可进入下一阶段，无需整组等待。 |
| 数量准入与背压 | 当前部署`--max-inflight 0`，不按请求总数拒绝，由显存准入控制；阶段队列满时已接收工作等待空间。兼容默认配置仍为8，可显式设置正数启用数量限制。 |
| 取消、异常与关闭 | 请求异常反馈到对应请求；HTTP等待取消后，已接收工作继续执行并占用容量；服务关闭前排空请求和阶段队列。 |
| 设备适配 | 设备选择、可见性检查与初始化通过`dev/device-context` runtime；底层Torch、厂商扩展、驱动及运行库由平台环境提供。 |
| 设备内存探测 | 通过`runtime.memory_stats()`后台读取总量／已用／可用内存，默认每秒采样；`/health.device_memory`暴露缓存读数、时间、年龄、过期标记及错误状态。 |
| 内存准入原型 | 显式开启后按高低水位、安全余量及每请求全链路预算接收请求；余量不足或读数不可用返回503，回落后恢复；请求预算保留到处理结束并经新采样确认后释放，数量上限仍有效。 |
| 可观察性 | `/health`展示真实数据库、检索模式、在途数量、阶段队列状态及设备内存；服务端日志记录请求ID和阶段排队／执行时间。 |
| 数据库部署 | Milvus采用内嵌etcd的单容器部署；Elasticsearch单容器部署；数据通过RAID目录持久化。 |

阶段worker默认为：向量化2个HTTP worker、检索4个worker、精排1个worker；每个阶段8个等待槽。
本地模型在其初始化线程执行，内存探测使用独立采样线程。向量化worker并发指向vLLM提交请求，
不等同于已验证这些请求进入同一次模型forward。

每个请求的查询、向量、候选与结果目前通过参数和局部变量传递，采用固定阶段依赖；
尚未抽象为通用流程上下文或动态工作流引擎。

## 已保存的验证证据

| 日期 | 验证范围 | 结果与证据 |
|---|---|---|
| 2026-10-28 | 更新后的device-context接口兼容性 | 实际RAG/runtime代码配模拟厂商接口，7项专项检查通过；Ascend离线自检92通过、0失败、2跳过。现有设备初始化与内存探测调用可保持；容器已停止，本次未做NPU实测。[报告](artifacts/checks/device_context_compatibility/2026-10-28/report.md)。 |
| 2026-10-09 | 双库全量重建与持久化 | 两个后端各6,543块，全量ID及向量核对通过；Milvus重启后数据保留。[重建记录](artifacts/rebuilds/vllm-rebuild-2026-10-09/manifest.json)、[重启后验证](artifacts/rebuilds/vllm-rebuild-2026-10-09/post_restart_verification.json)。 |
| 2026-10-09 | 检索阶段顺序／并发比较 | 固定50条查询×两数据库×两模式，共200次检索成功，两库各50/50排序一致；测量前预计算向量，不包含模型阶段。[检索报告](artifacts/rebuilds/vllm-rebuild-2026-10-09/comparison-20261009T045201Z-2a44ec/report.md)。 |
| 2026-10-09 | 独立队列完整API链路 | 同一50条查询×两库×两种配置，共200次完整向量化→检索→精排调用全部成功；两库各50/50有序结果、内容及容差内分数一致。[完整链路报告](artifacts/benchmarks/queued-pipeline-2026-10-09_134251-78a48d/report.md)。 |
| 2026-10-09 | 同时／错峰、超限及恢复 | 同时8请求、错峰8请求均成功且匹配串行参考；9请求突发接受8个、预期拒绝1个，之后恢复成功；本轮34次调用中33次成功、1次预期503。[简要报告](artifacts/checks/queue_evidence/2026-10-09-summary/report.md)、[详细证据](artifacts/checks/queue_evidence/2026-10-09_140744-milvus-30d953/report.md)。 |
| 2026-10-09 | 跨请求阶段推进 | 33个成功请求均有完整三阶段记录且自身依赖顺序正确；观察到60对跨请求阶段区间重叠。[服务端时间线](artifacts/checks/queue_evidence/2026-10-09_140744-milvus-30d953/stage_timeline.csv)。 |
| 2026-10-14 | 设备内存及跨进程占用 | NPU总量62,748MiB，初始化后单次读取约0.7–1.2ms；独立进程分配128MiB时已用158→348MiB，退出后回到158MiB，增量包含运行环境开销。[单文件实测报告](artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。 |
| 2026-10-14 | 真实API后台内存采样 | 173次health检查均得到正常且未过期的缓存读数，采样时间戳持续刷新；2条串行及同样2条并发查询全部成功，结果一致。这173次是HTTP检查次数，并非173次独立设备采样。[实测报告](artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。 |
| 2026-10-14 | 完整回归 | `python -m pytest tests -q`：120项通过，覆盖设备路由、队列、API、内存指标校验、采样失败恢复及线程隔离等。[回归记录](artifacts/checks/memory_probe/2026-10-14_132823-f0ad80/report.md)。 |
| 2026-10-20 | 内存准入与完整回归 | 169项通过，覆盖高水位拒绝、低水位恢复、8请求同时到达仅2个预算可容纳时接受2个/拒绝6个、断开与失败后的预算持有/释放、探测异常及过期。采用受控读数与假模型，未做真实NPU压力或双库200次复测。[单文件报告](artifacts/checks/memory_admission/2026-10-20/report.md)。 |

固定输入是[同一组50条NFCorpus查询](data/queries/nfcorpus-50.jsonl)，200次调用不代表200条不同查询。
有序结果比较采用分数相对容差1e-4、绝对容差1e-6；一致性验证不等于基于qrels的检索质量评估。
完整API的两种配置同时改变了检索模式与客户端提交并发度，不能单独据此计算新增队列的开销。
该轮未观察到明显吞吐提升，不能将阶段重叠表述为已取得吞吐加速。

vLLM曾出现相同输入的向量不稳定问题，当前启动器为embedding进程设置
`TRITON_ENABLE_TASKQUEUE=true`；固定输入及pipeline控制实验已保存。
定位记录见[诊断结果](artifacts/diagnostics/vllm-mismatch-diagnosis-2026-10-09/summary.json)。

## 当前运行配置与依赖

以下是部署配置；容器和服务的实时在线状态通过health检查确认。

| 项目 | 当前配置 |
|---|---|
| 开发容器 | `rag-ljy-vllm-910c`；2026-10-29配置改为 `/dev/davinci1`（die 1，物理卡0/chip 1），容器内逻辑`npu:0`。已有容器仍为 `/dev/davinci7`，待重建生效；本次未启动或重建。 |
| vLLM embedding服务 | `http://127.0.0.1:8089`；独立模型进程。 |
| RAG查询API | `http://127.0.0.1:8088`；入口`POST /query`，状态`GET /health`。 |
| Elasticsearch | `rag-ljy-elasticsearch`，宿主机`127.0.0.1:9201`映射到容器9200；索引`nfcorpus-chunks-v1`。 |
| Milvus | `rag-ljy-milvus`，`127.0.0.1:19530`；集合`nfcorpus_chunks_v1`。 |
| 计算分工 | 两个数据库的BM25／dense检索均在CPU执行；向量化与精排模型在NPU执行。 |
| device-context | 目前通过`PYTHONPATH`加载`dev/device-context/prototype`，用于设备初始化与内存探测。 |
| 推理环境 | 当前昇腾环境依赖镜像中的PyTorch／torch_npu、vLLM-Ascend及匹配驱动／运行库；跨平台完整验证尚未完成。 |
| Python打包 | 已有`pyproject.toml`，包名`flagos-rag-ljy`；独立安装交付的命令入口与依赖验收仍待完善。 |

启动时须使用现有NFCorpus索引／集合名称；操作命令、环境变量和端口说明见[运行指南](docs/operations.md)。

## 内存准入规则更新（2026-10-20）

- 当前代码：`实测占用 + 已有预留 > 85% × 总内存` 时进入暂停，回落至75%或以下解除暂停；未暂停时，只有 `实测空闲 − 已有预留 − 新请求预算 ≥ 1024MiB` 才接受请求（水位和余量均可配置）。不再要求加入新请求预算后仍低于85%。
- 更新后完整回归173项通过，包含高水位严格大于边界、安全余量等号边界、允许新预算跨过高水位后暂停后续接收、预算不足不触发低水位锁定。[验证报告](artifacts/checks/memory_admission/2026-10-20-rule-update/report.md)。采用受控读数；尚未验证真实NPU拒绝与恢复。
- 本次未重启正在运行的API；需重启查询服务才能加载新规则，embedding服务无需重启。

## 当前限制与下一步

- **内存准入为可选原型，默认不启用。** 开启须显式配置每请求预算；目前采用全链路固定预算记账，非真实内存预分配，尚未按长查询、并发模型峰值校准，也未接入主机内存限制或OOM后模型恢复。
- 内存接口统计范围仍标记`backend_reported_unverified`；默认准入拒绝这类读数，验证环境后可显式允许。Ascend兼容回退可能只反映当前进程分配量，既有跨进程实测不等于对全部后端的保证。
- 下一步在真实NPU负载下校准预算与高低水位，并复测长查询、并发、内存压力与回落恢复；2026-10-20受控测试不等同于真机防OOM验收。当前默认水位85%/75%、安全余量1024MiB，预算及采样间隔可配置。
- 尚未验证vLLM实际forward合批，尚未实现跨请求精排合批或vLLM精排后端；当前单请求候选batch不等同于跨请求batch。
- 阶段队列为进程内存对象，暂无重启后的请求恢复或跨进程任务分发；请求截止时间、持续负载及更多故障场景仍需完善。
- 尚未完成NFCorpus qrels的nDCG／MRR／Recall质量评测；中文analyzer配置及其重建验证待按需求开展。
- 生成模型尚未接入，目前API输出为精排后的检索片段。
- 独立pip安装包的交付验证待完成，设备运行环境、模型与数据库仍需分别准备。

## 实现与文档入口

| 内容 | 入口 |
|---|---|
| 阶段队列与请求调度 | [scheduler.py](src/rag_engine/scheduler.py)、[server.py](src/rag_engine/server.py) |
| 设备适配与内存采样 | [device.py](src/rag_engine/device.py)、[memory_probe.py](src/rag_engine/memory_probe.py) |
| 内存准入与预算 | [admission.py](src/rag_engine/admission.py)、[受控验证](tests/test_admission.py) |
| 内存探测复跑 | [check_memory.py](scripts/checks/check_memory.py) |
| 服务启动 | [serve_embeddings.py](scripts/runtime/serve_embeddings.py)、[serve_queries.py](scripts/runtime/serve_queries.py) |
| 结构、操作与评测 | [实现结构](docs/architecture.md)、[运行指南](docs/operations.md)、[评测指南](docs/evaluation.md)、[device-context接入](docs/device-context.md) |
| 测试与原始证据 | [结果索引](artifacts/README.md)、[机器可读索引](artifacts/index.json) |

## 接手可执行性核查（2026-10-29 16:44，Asia/Shanghai）

本次仅核查仓库、容器进程、已有结果和实时health，未启动新压力测试，未重新运行回归。

- `docker inspect`确认运行中的`rag-ljy-vllm-910c`已映射宿主机`/dev/davinci1`；上文“仍为davinci7、待重建”是此前状态，已被此次实查取代。
- `run_pressure_suite.py`与Elasticsearch的`check_online_pressure.py`仍在运行；自动调度后续计划恢复Milvus再执行显存专项。不得同时启动第二套压力程序。调度记录见[当前目录](experiment_outputs/checks/pressure_suite/20261029T081319/)。
- 16:44:26的设备采样：总量62,748MiB、已用13,906MiB（22.16%）、空闲48,841MiB。health显示Elasticsearch、8个在途请求、18,432MiB未释放预算，有效占用51.538%；显存准入拒绝计数为0。此快照不提供NPU计算利用率。
- [Milvus结果](experiment_outputs/benchmarks/20261029T080521-milvus-pressure/summary.json)为请求负载测试：3,103次成功、2,409次数量容量拒绝；不能作为85%显存准入验证。此次核查未发现显存专项结果，不能宣称已执行或通过。
- 准入实现包含85%严格高水位、75%恢复、安全余量、过期读数拒绝，以及完成后等待新采样释放预算。现有[真机脚本](scripts/checks/check_memory_admission_live.py)计划额外分配显存至85.7%、保留8GiB保护余量、80%验证滞回、释放后恢复；压力来源是额外分配，不是查询自然占用。
- 执行前仍需完善验收证据：脚本只保存自身快照，尚未完整固化代码/模型版本和输入；未持续记录所有等待阶段的显存时间线；`pressure_drain_overlap`仅记录而不要求为真，不能保证已证明压力期间在途请求继续完成；清理函数或最终health失败可能阻止摘要落盘，需要独立容错及超时/退出清理。
- 预算拒绝子测试使用临时60,000MiB预算，并非生产2GiB预算。以当前62,748MiB总量、85%高水位计算，未触发高水位时可用余量足以覆盖2GiB预算和1GiB安全余量；该拒绝分支需要明确标注的测试配置，不能声称是当前2GiB配置自然触发。

可继续开展真机验证，但上述运行状态与证据缺口必须先处理；通过该实验也不等于已校准2GiB预算或保证任意负载不会OOM。

### 停止请求后的核查（2026-10-29 16:46，Asia/Shanghai）

用户要求停止当前测试。实际检查时，请求压测、自动调度及显存专项进程均已自行退出，无需发送终止信号，未启动任何新测试。调度日志记录Elasticsearch退出码0、显存专项退出码1；显存专项已有[结果文件](experiment_outputs/checks/memory_admission/20261029T084549-real-pressure/summary.json)，本次未进一步验收，不应视为整体通过。上节16:44的“尚未发现专项结果”仅代表当时状态。

容器及模型、数据库服务保留运行。16:46:36的health显示API已恢复Milvus、ready、在途请求0、预留预算0、显存占用21.1%。当前没有继续运行的测试或自动调度进程。

### 三模式水位实验代码准备（2026-10-29）

- 新增[实验脚本](scripts/checks/check_watermarks_modes.py)和[启动说明](docs/watermarks-experiment.md)：同时突发、错峰、持续到达分别覆盖基线、真实85%以上、约80%滞回、释放恢复。默认仅预检查，必须显式`--run`才能施加压力，不启动旧自动调度或辅助API。
- 额外分配按128MiB步进，8GiB显存保护余量；支持SIGINT/SIGTERM清理、运行时限和服务恢复检查。记录实际设备读数、API采样时间、预算、请求完整HTTP响应、输入、源代码和模型文件哈希；输出保存在外部experiments目录。
- 数量上限503不能作为水位证据；中间阶段同时检查实测与有效占用；未观察到压力期间在途工作时标为incomplete，不虚报通过。预算不足、过期读数及NPU计算利用率不属于此脚本的验证范围。
- 7项离线证据判定测试通过；2026-10-29 17:07（Asia/Shanghai）容器内默认预检查通过，Milvus服务空闲，显存21.1%。本轮没有执行新脚本的真实显存压力实验，尚无三模式真机通过结论。
- 已读旧专项摘要：失败原因为`OSError(98, 'Address already in use')`；旧摘要记录部分水位及预算检查成功，但不替代新三模式验收。

### timeout启动器误报修复（2026-10-29）

用户使用文档中的timeout启动命令时，进程检查将自身父进程的脚本参数误判为另一份实验，在分配显存前退出。现已仅排除自身祖先链中的启动器，仍检测其他实验及祖先调度脚本，互斥锁保持有效。8项离线测试通过（含自身timeout、另一份timeout及调度脚本的回归场景）；18:52（Asia/Shanghai）容器内使用相同timeout包装、不带`--run`的预检查通过。本次未执行压力实验，原启动命令可继续使用。

### 用户启动后的运行核查（2026-10-29 18:54～18:55，Asia/Shanghai）

用户已启动三模式实验，输出目录[20261029T105348.351974Z](experiment_outputs/checks/watermarks_modes/20261029T105348.351974Z/)。进程与events持续更新，并非卡住；18:54:40 API采样实测53,746/62,748MiB（85.65%），准入paused/memory_pressure。18:55已进入burst的recovery阶段，采样显存22.26%、准入ready。当前仅是运行中观测，尚未验收整轮或三模式最终结果。

原脚本缺少终端进度提示，已补充准备/哈希/阶段/每5秒health/清理提示，8项离线测试通过；正在运行的旧进程仍写events，不需要因此重启。宿主机`/usr/local/bin/npu-smi info`实测可用，容器内PATH及常见驱动工具路径未找到该命令。API的max_inflight=8是配置的在途请求上限，不是设备容量或NPU利用率。

### 三模式实验完成核查（2026-10-29 19:02，Asia/Shanghai）

[本轮摘要](experiment_outputs/checks/watermarks_modes/20261029T105348.351974Z/summary.json)为passed：burst、staggered、continuous各自的基线/高位/中间段/恢复共12项检查通过，各模式均记录压力暂停与在途请求重叠。89次HTTP 200、67次HTTP 503，无请求传输错误；原始events复核67次拒绝均为memory_pressure。压力由检查进程额外分配产生，不是查询自然占用。

实验进程已退出；19:02实时health确认Milvus ready、inflight=0、reserved_mib=0，显存22.16%。此结论限于本轮三种到达模式的高低水位规则，不覆盖预算校准、预算不足分支、过期读数分支、NPU计算利用率或普遍OOM安全保证。

### 取消8请求入口上限与真实请求实验修订（2026-10-29 20:07，Asia/Shanghai）

- 按用户要求新增`--max-inflight 0`并实际重启查询API生效，health确认max_inflight=0、ready；embedding服务和数据库未重启。保持85%/75%水位、2GiB请求预算和1GiB安全余量。配置前后与日志见[变更记录](experiment_outputs/checks/count_limit_change/20261029T120704Z/)。
- 76项API/显存准入/调度回归通过，包括同时接受10个请求、显存压力拒绝、完成后恢复。阶段等待槽8不再构成请求入口拒绝条件，模型执行并发仍是embedding 2、rerank 1；没有把在途数当作实际NPU执行并发。
- 新[真实请求脚本](scripts/checks/check_request_memory_modes.py)通过burst/staggered/continuous递增实际`/query`请求及受支持的精排batch，禁止直接分配设备显存、降低阈值或清缓存制造恢复。5项证据判定测试通过，运行API上的只读预检查通过；未启动新版压力实验。
- 默认原始50条查询仅4～53字符，不代表长文本负载。脚本接受用户提供的真实长查询输入，不拼接填充内容。实际占用未超过85%时明确记录threshold_not_reached；预留预算跨85%不能代替实际占用。中间滞回窗口未自然出现时标incomplete。
- 两个旧额外分配实验入口已禁止启动；历史证据保留并更正范围。新版启动方式见[操作说明](docs/watermarks-experiment.md)。没有提交或推送Git。

### 全量输入与真实推理校准（2026-10-29 20:24～20:31，Asia/Shanghai）

- 已实际准备[3,237条原始查询+64条语料上下文压力请求](experiment_outputs/inputs/request_pressure/manifest.json)，长输入为1,024/2,048/4,096/6,144 token，每条保留原查询、文档来源和构造标识。不是声称存在64条原始长人类查询，也不是空张量显存压力。
- 新增可配置精排forward batch上限并部署32，embedding提交worker从2升到8，设备采样改为0.1秒。max_inflight=0及85%/75%、2048MiB预算、1024MiB余量不变。[配置与服务日志](experiment_outputs/checks/request_pressure_configuration/20261029T122149Z/)。77项API/准入/调度测试通过。
- [首次真实校准](experiment_outputs/checks/request_memory_modes/20261029T122406.038035Z/summary.json)：3条6,144-token请求，精排batch依次1/2/3，全部HTTP 200，实测显存升到77.411%。客户端错误要求每个递增档位都恢复至75%以下而提前停止，已修复；该轮保留failed，不改写历史结果。
- [继续校准](experiment_outputs/checks/request_memory_modes/20261029T122702.736228Z/summary.json)：第4条请求精排batch4，根据历史增量将输入截短为4,943 token。实测先达到87.27%且memory_pressure，峰值97.17%，最少空闲1,776MiB。请求最终HTTP 200，服务日志未检出OOM异常，但已违反8GiB保护线，故状态failed。
- 两轮共4条真实RAG请求，包含实际embedding、Milvus检索、Transformers精排；没有单独的NPU分配程序。模型实际分配/缓存增长具有非线性，前几步线性预测不能保证峰值安全，当前2GiB请求预算不足以保护这种负载。不能把这次越过85%报告为防OOM或三模式验收通过。
- 检查进程已退出；20:30后实查API ready、inflight=0、reserved_mib=0、显存约52.43%。新增[safety_hold.json](experiment_outputs/checks/request_memory_modes/safety_hold.json)阻止原加压方案再次自动执行。后续必须修正并验证单次forward峰值保护，再完成三模式拒绝/滞回/恢复；不通过删除暂停记录重复冒险。
- 新脚本5项证据分类测试通过，源文件编译检查通过；未提交或推送Git。

### 逐请求动态估算与执行保护（2026-10-29 21:24～21:26，Asia/Shanghai）

- 已部署`--memory-budget-mode dynamic`，health中`request_budget_mib=null`、`estimate_requests=true`、`max_inflight=0`。每条请求在进入模型/阶段队列前按实际token、有效batch、Qwen3结构、实际dtype与执行配置估算embedding及rerank新增峰值；预算分别登记并保留至完成后的新采样。不再假设所有请求占用2GiB。[当前部署记录](experiment_outputs/checks/dynamic_memory_admission/20261029T132408Z/)及[目标/范围](docs/memory-admission-requirements.md)。
- 增加CPU只读`POST /memory/estimate`；真正`/query`重新估算并检查预算，成功响应头记录估算值。未知执行配置、无效估算或估算故障不回退固定预算。当前支持同容器可核对的本地vLLM Qwen3 embedding与Transformers Qwen3 reranker，显存读数仍来自device-context。
- 精排每次forward前按实际padding形状及最新显存复核；必要时缩小实际batch、回收未使用的自身缓存后复核。记录请求ID、Torch分配器峰值、token/batch及执行指纹，观测超过估算会锁定故障并停止新准入。此分配器观测不覆盖远端embedding进程或分配器之外的内存，也不能用请求结束后的采样代替所有瞬时峰值。
- 文档上界为1,024token，离线核对当前6,543块最大578token，当前语料不被该限制截断。未来超限文档会被截断并记录。
- [最新受限规模真机检查](experiment_outputs/checks/dynamic_estimator_live/20261029T132544Z/summary.json)：8条真实RAG请求HTTP 200，覆盖短查询和1,024/2,048/4,096-token上下文、batch1/2/4的部分组合。动态估算分别为2,639～28,916MiB；各次执行前均要求预估至少剩余8GiB。已知危险的4,943-token/batch4请求估算81,718MiB，返回503/request_budget和Retry-After，接受计数不增加，不进入模型。此503证明预算不足拒绝，不能当作85%高水位拒绝。
- [峰值对比](experiment_outputs/checks/estimate_calibration/20261029T132614Z/summary.json)：上述8条请求产生166次forward，所测分配器新增峰值均未超过对应forward估算，最大比值约0.2935。解析模型仍标为`analytical_uncalibrated`，保持2倍系数及每阶段512MiB工作区，不根据这几条数据擅自缩减系数，也不宣称任意长度/batch/并发下均有安全上界。
- 全部207项回归通过。最新检查结束后API ready、inflight=0、reserved_mib=0、估算fault为空，设备已用26,977/62,748MiB（42.99%）。21:24的API重启用于部署及建立校准基线，不作为低水位恢复证据。
- 三模式实际85%拒绝/滞回/恢复、并发归因与更大输入/冷热缓存覆盖仍未完成；旧真实压力脚本的`safety_hold.json`保留。两个额外分配实验入口仍禁止执行。所有本轮数据落在外部experiments目录，未提交或推送Git。

后续功能、配置或验证结果发生变化时，直接更新本文件，并链接对应的实现与证据。
