# Dedicated stage queues: 200 full-pipeline API cases

Date: 2026-10-09 (Asia/Shanghai). Physical NPU 7, logical npu:0.

Same frozen 50 NFCorpus queries × two databases × two modes. All 200 HTTP requests succeeded; all 200 requests have complete stage timing records and unique request IDs. Each database has 50/50 matching ordered results and scores under the existing relative 1e-4 / absolute 1e-6 tolerance.

Both modes use dedicated queues. Sequential mode sends one request at a time and uses sequential BM25/dense; concurrent mode sends waves of two and uses concurrent BM25/dense. This comparison cannot isolate the cost of adding queues. Embedding and reranking are included; the old 200-case baseline measured retrieval only.

| Database | Mode | Mean request latency (ms) | P95 (ms) | Observed requests/s | Embedding overlap pairs |
|---|---|---|---|---|---|
| milvus | sequential | 447.71 | 597.39 | 2.233 | 0 |
| milvus | concurrent | 688.08 | 1024.05 | 2.248 | 25 |
| elasticsearch | sequential | 437.18 | 606.82 | 2.287 | 0 |
| elasticsearch | concurrent | 685.06 | 1128.60 | 2.260 | 25 |

Concurrent submission increases each request’s waiting time in this run; throughput and per-request latency are different measurements. Shared-host single-pass measurements are not a statistically robust capacity test.

Queue defaults: 8 waiting slots per stage, 8 total admitted requests; embedding=2, retrieval=4, rerank=1 workers. Full stage queues backpressure admitted work; overall admission full returns HTTP 503 with Retry-After. There is no deliberate batch-fill wait.

Stage logs prove overlapping embedding calls. They do not establish that vLLM combined those calls into one forward pass. Cross-request reranking batching and memory-aware admission are not implemented.

Artifacts: per-backend/per-mode responses.json, health.json, stages.json, server.log; summary.json; immutable source snapshots below.
