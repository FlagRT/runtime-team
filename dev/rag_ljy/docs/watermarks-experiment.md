# 真实请求驱动的85%显存验收

最新状态以根目录[status.md](../status.md)为准。2026-10-30北京时间已完成三个独立冷启动实验和读数过期检查：[审计汇总](../experiment_outputs/checks/memory_acceptance/20261029T162525Z/summary.json)、[实际显存与软件预算时间线](../experiment_outputs/checks/memory_acceptance/20261029T162525Z/memory_timelines.svg)。

| 模式 | 正式负载到达 | 实测峰值 | 高位拒绝 / 中间滞回 / 低位恢复 / 在途完成 |
|---|---|---:|---|
| 同时突发 | 2条同步释放，实测间隔0.629ms | 85.627% | 全部通过 |
| 错峰到达 | 2条，实测间隔0.211s | 85.627% | 全部通过 |
| 持续到达 | 5QPS开放循环，达到高位前提交8条 | 85.627% | 全部通过 |

采样最低空闲9,019MiB，三轮结束时18.39%、inflight=0、预留=0、估算fault为空。全部拒绝带503及Retry-After；预算拒绝与memory_pressure单独统计。NPU计算利用率未测。这里验证Milvus链路，不声称Elasticsearch完成同样专项。

## 实际策略与压力来源

- 唯一设备指标：`device-context.runtime.memory_stats()`。实测≥85%暂停，暂停后只有≤75%才恢复；软件预留不参与水位触发。
- 新请求必须满足`实测空闲 − 未释放预算 − 新增动态估算 ≥ 5120MiB`。本轮51.685%时实际出现预算拒绝：30,317−24,005−3,638=2,674MiB，不够5GiB；这不算高位拒绝。
- 预算按token、dtype、模型、实际执行计划及已验证的缓存可复用性计算，非固定2GiB。接口`POST /memory/estimate`仅预览；`/query`重新原子检查和预留。不得用预算上调伪造85%。
- 每条压力请求完整执行vLLM embedding、真实Milvus检索、Transformers rerank。精排候选50、返回5条；请求batch上限32，本轮准入计划30，执行前仍根据实际padding形状复核。
- 显存来自这些真实推理及它们留下的allocator缓存，主要是reranker的计算/输出张量。无额外压力分配、无扩大模型常驻池、无阈值下调。服务自动回收闲置压力缓存，客户端不清缓存。
- 8请求入口限制已关闭：`max_inflight=0`。阶段队列8槽只是已接收工作的等待容量；rerank仍只有一个模型worker。三种模式描述HTTP到达方式，不是三种NPU计算并行方式。

## 为什么使用独立冷启动实验

每轮开始前重启已排空的API，统一初始条件；每轮内部从高位、经过中间区间、到恢复完成均保持同一API/采样进程。准备阶段重启不计作恢复证据。

每轮1条短请求初始化、2条相同约1,024-token真实请求预热，然后正式提交约2,048-token压力输入。突发与错峰各2条负载（各接受1条），持续到达8条负载（接受1条），其余为预算拒绝；随后在高位及释放阶段发送独立真实探测请求。峰值时各轮只有1条RAG在途。累计HTTP为20/20/25条，包括初始化、预热、负载和探测，不能说成这些请求同时占用显存。

请求文本来自已保存的NFCorpus语料上下文压力输入，明确标记为构造测试输入，不冒称原始人工查询。完整输入来源、模型哈希、源码、每条payload与HTTP时间都在每轮目录内。

同一profile在前一轮回收后的暖缓存布局下曾仅达到84.18%～84.20%，旧记录仍为failed/incomplete。因此不能声称任意缓存历史下固定几条请求必定达到85%。本验收覆盖上述独立初始条件；解析估算亦不等于任意模型/输入的通用上界。

## 在当前宿主机复现

默认仅预览，不重启、不发查询：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_measured_acceptance.sh
```

执行完整三轮及采样过期检查（会在**每轮开始前**重启已排空的API）：

```bash
bash scripts/checks/run_measured_acceptance.sh --run
```

该脚本遇到失败即停止下一轮。每步打印`OUTPUT=...`与`RESULT=...`；只有完整通过返回0。不要同时运行旧负载调度器。Ctrl-C停止新增提交，已接收请求由服务继续完成，随后检查`/health`中的inflight与reserved均归零。

单轮示例，必须先准备独立初始条件：

```bash
docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python scripts/runtime/restart_memory_api.py --run
docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python -u scripts/checks/check_measured_watermarks.py --run \
  --modes burst --count 2 --warmup-profile 1024:32 --profiles 2048:32
```

替换`burst`为`staggered`或`continuous`。直接在一个进程中连续跑三个模式是另一种暖缓存实验，不能假设它等价于这里已通过的独立实验。

## 监测与构建

当前部署的采样间隔为0.001秒，采用独立进程调用同一个device-context接口，避免模型allocator锁阻塞采样；实际采样时延与年龄另存，不能认为每1ms必定采到。观察进程不分配压力tensor，持久上下文开销计入空载基线。读数年龄包含读操作耗时，超过3秒拒绝。

服务缓存回收通过公开C++ allocator接口执行，CDLL释放GIL，使API在回收时仍可拒绝请求；不添加回收延迟、不人工维持80%。桥接会验证torch、torch_npu、ABI、源码及二进制哈希。当前桥接已构建；更换运行时后需重建：

```bash
docker exec -w /workspace/dev/rag_ljy rag-ljy-vllm-910c \
  .venv/bin/python scripts/runtime/build_npu_cache_bridge.py
```

将打印的库路径传给准备工具`--cache-bridge /path/to/libnpu_cache_bridge.so`。构建产物与编译命令在`experiment_outputs/runtime_extensions/`。当前环境配置：`RAG_PROFILE_FIRST_FORWARD=1`、`RAG_MEMORY_PROBE_PROCESS=1`、`RAG_RERANK_DOCUMENT_TOKENS=640`、`PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.70`、allocator fraction 0.99；完整启动参数在每次部署记录中。

## 结果目录与历史失败

所有输出经仓库`experiment_outputs`符号链接保存到：

`/home/jliu171/jliu171_data/hetero_data/rag/experiments/`

- `checks/measured_watermarks/<UTC>/`：manifest、源码/模型哈希、输入、逐请求与实测时间线、判定。
- `checks/dynamic_memory_admission/<UTC>/`：准备阶段前后配置、非敏感运行环境、API日志。
- `checks/stale_memory/<UTC>/`：仅暂停采样进程的故障注入、503及恢复200。finally和独立8秒恢复看门狗保证SIGCONT；不暂停模型进程。
- `checks/memory_acceptance/<UTC>/`：三轮审计汇总、部署日志副本、代码哈希、Git基线及未提交修改状态。汇总工具`summarize_measured_acceptance.py`，绘图工具`plot_measured_acceptance.py`。

旧97.17%越过保护线的实验仍为failed，旧450次请求最高44.629%仍未达到阈值，外部分配实验不计入本验收。旧`check_request_memory_modes.py`的安全暂停记录保持；当前入口使用新的逐forward保护，不删除历史失败后照旧重跑。

## Elasticsearch入口（2026-11-04，压力验收待用户执行）

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_elasticsearch_memory_acceptance.sh --run
```

省略`--run`仅检查索引和预览配置。脚本显式切换`RETRIEVAL_BACKEND=elasticsearch`，逐轮冷启动后运行相同三模式及采样过期检查，实际health后端不匹配即停止。索引预检查确认向量映射、3,633文档/6,543分块及640-token文档上界，保存语料指纹。服务正常回收完成滞回恢复，不以准备阶段重启作恢复证据。

默认profile与Milvus相同，但Elasticsearch检索候选不同，实际峰值必须重新验证。未达85%或缺少中间滞回等证据会返回非零，停止套件，不生成通过汇总。需要调整真实输入档位或请求batch时可显式传`--profiles 2048:32`和`--warmup-profile 1024:32`；不得更改水位或使用额外分配冒充请求压力。

新结果分别位于`checks/measured_watermarks/elasticsearch/`等按后端分隔的目录；全部检查通过后自动调用汇总工具。运行结束API保留Elasticsearch后端。只读预检查已通过，尚未执行Elasticsearch高水位压力实验。

### Elasticsearch长查询子句超限修复（2026-11-04）

首次用户运行在两条预热请求的BM25阶段返回500，原始ES错误为`maxClauseCount=1024`；没有达到高水位。当前仅对子句超限400进行一次有界词法回退：各字段analyzer产生的词项去重后各取前256项，构造最多512个term子句的dis_max，title boost保留2。普通查询仍用原multi_match，其他错误不隐藏。完整查询仍进入embedding/rerank；有界词法回退可能改变候选排序，需与显存验收结果分别理解。

预检查现包含64条已保存长输入的真实CPU词法检索；已通过该检查和234项回归，未重跑NPU压力。启动命令不变，套件开始前重启API会加载新代码。失败证据和修复验证位置见根目录status.md。

## Elasticsearch已通过的最终配置（2026-11-04）

用户使用`bash scripts/checks/run_elasticsearch_memory_acceptance.sh --run --profiles 2112:32`完成三模式及采样过期检查，汇总[20261104T113704Z](../experiment_outputs/checks/memory_acceptance/elasticsearch/20261104T113704Z/summary.json)为passed。突发/错峰/持续实测峰值分别85.443%/85.442%/85.440%，每轮高位拒绝、中间滞回、低位恢复、在途完成均通过，最低采样空闲9,134MiB。复现已通过配置时须明确传入`--profiles 2112:32`；默认2048档曾只达到84.557%，此前失败及未达阈值记录不修改。上述较早“待用户执行”的表述为历史进度。
