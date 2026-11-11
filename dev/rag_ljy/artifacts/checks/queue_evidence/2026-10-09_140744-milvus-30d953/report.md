# 在线多请求与独立阶段队列：实测证据

日期：2026-10-09T14:07:44.874692+08:00（北京时间）；结论：**passed**。

在线 API 实际后端：milvus；embedding=vLLM；精排=Transformers。
容器 `rag-ljy-vllm-910c`，仅映射物理 NPU 7，在容器内使用逻辑 `npu:0`。
固定输入：NFCorpus 50 条查询文件的前 8 条；超限场景重复第一条，便于逐一对照串行参考。
查询文件 SHA256：`9b0a898ba4e50ccf0081591ba4a2a1ddccc960c112582692a52e90aca09e7429`。

| 验证内容 | 实测结果 |
|---|---|
| 串行参考 | 8/8 HTTP 200 |
| 同时到达 | 8/8 HTTP 200，全部匹配串行结果 |
| 错峰到达（间隔 20ms） | 8/8 HTTP 200，全部匹配串行结果 |
| 超限突发 9 请求 | 接受 8 个；1 个 HTTP 503，Retry-After=1；接受的结果全部匹配串行参考 |
| 超限后恢复 | 1/1 成功；最终 inflight 和三个队列均归零 |
| 阶段依赖及结果归属 | 33 个成功请求均有完整三阶段记录；自身 embedding→检索→精排依赖顺序正确；所有请求 ID 唯一 |
| 跨请求阶段推进 | 日志观察到 60 对检索/精排与其他请求 embedding 的阶段区间重叠 |
| 有界队列 | 525 次有效采样均未超限；在途峰值 8；等待峰值 embedding=6、retrieval=1、rerank=7 |

队列等待槽各 8，全局准入上限 8；worker 分别为 embedding=2、retrieval=4、rerank=1。
结果核对包含排名、块/文档 ID、正文、标题、来源与检索名次；分数容差为相对 1e-4 / 绝对 1e-6。

## 同时到达场景：各请求独立推进

以下毫秒数来自服务器 monotonic_ns，以本场景最早的 embedding 开始时刻为零点。

| Query ID | embedding 起止（ms） | 检索起止（ms） | 精排起止（ms） |
|---|---|---|---|
| PLAIN-1028 | 50.94–143.14 | 143.83–150.77 | 702.84–1007.28 |
| PLAIN-1130 | 279.61–369.91 | 370.50–375.54 | 2239.13–2540.41 |
| PLAIN-1141 | 0.00–50.49 | 51.15–59.01 | 59.29–397.46 |
| PLAIN-1161 | 0.61–94.50 | 94.87–101.59 | 397.76–702.53 |
| PLAIN-1172 | 143.62–233.99 | 234.45–240.94 | 1318.30–1624.89 |
| PLAIN-1225 | 187.48–279.21 | 279.80–286.38 | 1625.21–1933.29 |
| PLAIN-1409 | 94.75–186.97 | 187.56–195.53 | 1007.76–1317.98 |
| PLAIN-1441 | 234.38–324.14 | 324.58–330.39 | 1933.60–2238.79 |

例如 PLAIN-1141 在 59.29ms 已进入精排，其余请求最后一个 embedding 到 369.91ms 才结束，说明无需等待整组请求完成上一阶段。

## 原始证据及边界

- `serial.json`、`simultaneous.json`、`staggered.json`、`overload.json`、`recovery.json`：完整响应与请求 ID。
- `server.log`、`stage_timeline.json`、`stage_timeline.csv`：本轮服务端阶段记录与时间线。
- `health_samples.json`、`health_after.json`、`summary.json`：队列采样、恢复状态与判定。
- `source_snapshots/` 和 `SHA256SUMS`：本轮脚本及核心实现快照、原始输出校验。

本轮共 34 次 HTTP 调用：33 次成功、1 次预期超限拒绝；这是正确性与控制行为验证，不是吞吐量收益测试。
阶段区间包含 CPU/HTTP/NPU 等待，重叠证明调度层独立推进，不证明底层 NPU kernel 同时执行或 vLLM 将请求合并为同一 forward batch。
**基于内存水位的准入控制尚未实现**；本轮证明的是请求数量准入上限与三个独立有界队列。
stage queue 满槽后的背压、取消及失败隔离另由 `tests/test_scheduler.py` / `tests/test_server.py` 覆盖；本轮未把任何阶段队列填至 8 个等待项。
已有双库 200 次完整链路测试见 [报告](../../../benchmarks/queued-pipeline-2026-10-09_134251-78a48d/report.md)。

## 调度测试与复跑

22 项现有调度/API测试全部通过（0.50s），覆盖队列满槽背压、FIFO推进、结果归属、错误隔离、超限拒绝、取消及优雅退出。原始日志见 `scheduler-tests.log`，JUnit记录见 `scheduler-tests.xml`。

在容器内的 `/workspace/dev/rag_ljy`，激活 `.venv` 并确保8088 API空闲后执行：

```bash
python scripts/check_multi_request/capture_queue_evidence.py
python -m pytest tests/test_scheduler.py tests/test_server.py -q
```
