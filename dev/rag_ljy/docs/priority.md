# 请求优先级

`POST /query` 可选字段 `priority`：`high`、`normal`、`low`，默认 `normal`，其他值返回 HTTP 422。

```json
{"query":"What are the benefits of dietary fibre?","generate":true,"priority":"high","max_new_tokens":256}
```

向量化、检索、精排、生成各阶段独立按有效优先级出队；同级按进入该阶段队列的顺序执行。
默认在**该阶段队列内**每等待30秒提升一级，最高为high；低优先级等待60秒可提升至high。
进入下一阶段重新计时，不改变请求原始等级。启动参数 `--priority-aging-seconds 30` 可调整阈值，必须为有限正数。
这是等级提升时间，不是开始执行或响应时间保证。

只重排阶段队列内的工作，不抢占已派发工作，不保证不同阶段之间的全局优先顺序或最终响应顺序。
精排与生成仍共用原有模型执行线程，不能重排已提交给执行器的工作。
满队列不驱逐已接收请求，也不扩容；已接收请求等待空位，等待空位的生产者保留原来的背压顺序。
因此高优先级只有实际进入阶段队列后才参与优先排序。

所有优先级使用相同显存准入：实测85%暂停、75%恢复、新请求动态预算后保留5GiB。
高优先级不能绕过准入或全局请求数量限制（若配置）；被拒绝仍返回503和Retry-After，不内部重试。

`/health.scheduling` 提供策略和老化间隔；成功响应的 `X-Request-Priority` 返回原始等级。
日志 `event=dispatch` 记录请求ID、阶段、原始和有效优先级、队列等待时间。
当前完整vLLM工作流已通过统一启动入口加载优先级代码。

## 功能验证

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_priority_functional.sh
```

运行真实队列、线程及HTTP路由，使用模拟模型和模拟显存，无需重启在线API，不进行NPU推理。
覆盖三级排序、同级FIFO、三种到达方式、非抢占、30/60秒老化（注入时钟）、满队列背压、
HTTP四阶段等级传递、无效输入422，以及高优先级在显存高位/预算不足时仍拒绝。
三种到达方式是受控调度功能用例，不是持续负载性能验收。

输出在 `experiment_outputs/checks/priority_functional/<UTC时间>/`，即宿主机
`/home/jliu171/jliu171_data/hetero_data/rag/experiments/checks/priority_functional/<UTC时间>/`。
包含测试日志、JUnit报告及源码快照。真实NPU上的优先级负载、包含OneRec的85%/75%验收仍需另行运行。

## Elasticsearch原始50条：真实NPU生成测试

当前完整工作流已经接入vLLM并加载优先级代码，直接运行：

```bash
cd /home/jliu171/runtime-team/dev/rag_ljy
bash scripts/checks/run_priority_generation_50.sh --run
```

容器停止后使用`docker start rag-ljy-vllm-910c`自动恢复三项服务；等待日志`WORKFLOW=ready`。

脚本默认每批并发4条，50个原始查询各提交一次，依原文件顺序分配low/normal/high（17/17/16条）。
不修改查询文本，不重复请求，不改变服务显存预算。默认输出上限256tokens，与此前50条实验一致。
`--parallel 1`可用于串行生成覆盖，但不能用于证明优先级插队。
只做启动前检查可省略`--run`；测试入口不会自行重启服务。

输入：`data/queries/nfcorpus-50.jsonl`（固定SHA256校验）。
输出：`experiment_outputs/benchmarks/priority_generation_50/elasticsearch/<UTC时间>/`，对应宿主机
`/home/jliu171/jliu171_data/hetero_data/rag/experiments/benchmarks/priority_generation_50/elasticsearch/`。
包含`generated_responses.txt`可读回答、`queries.jsonl`原始输入、`requests.jsonl`完整请求响应、
`memory.jsonl`实际device-context读数、`summary.json`、版本与配置清单以及源码快照。

生成响应新增`scheduling.stages`：保存四阶段出队时原始/有效等级、等待时长、候选数及超越较早低等级请求的数量。
这是实际出队记录；不使用HTTP完成先后来推断插队。它不提供跨阶段全局排序，也不保证高优先级更早完成。

`GENERATION_COVERAGE=passed`仅表示50条均成功生成。
`PRIORITY_OVERTAKING=not_observed`或`PRIORITY_AGING=not_observed`表示未取得对应证据，不能算完整优先级验收通过。
保守预算可能只容许一个生成请求，其他请求返回503：照实记录为覆盖不完整，不重试，不放宽5GiB余量。
此实验也不替代85%/75%水位验收或回答质量评估；NPU计算利用率未采集。


当前8088同时启用Milvus和Elasticsearch。`/query`可填写`retrieval_backend`按请求选库，
不填写时使用默认库；两类请求共享优先级队列和显存准入。50条脚本使用
`--backend milvus`或`--backend elasticsearch`显式选库，并核对每条响应后端，不需要重建容器。
