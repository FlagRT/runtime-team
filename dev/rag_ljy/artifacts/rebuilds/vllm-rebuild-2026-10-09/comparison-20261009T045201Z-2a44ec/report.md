# Retrieval comparison

Run started (UTC): 2026-10-09T04:52:01.020087+00:00

50 identical queries; 1 measured pass(es) per mode; 3 excluded warm-ups per mode. Embeddings computed once; no reranking.

Each database runs separately. Requests are serial; concurrent means BM25 and dense overlap within one request. All times are client-observed milliseconds.

| Database | Mode | Success/fail | BM25 mean | Dense mean | Search mean | Search P50 | Search P95 | RRF mean | Total mean |
|---|---|---|---|---|---|---|---|---|---|
| milvus | sequential | 50/0 | 3.371 | 3.746 | 7.126 | 7.057 | 8.521 | 0.093 | 7.248 |
| milvus | concurrent | 50/0 | 3.368 | 3.904 | 4.246 | 4.242 | 4.732 | 0.089 | 4.365 |
| elasticsearch | sequential | 50/0 | 7.017 | 11.628 | 18.653 | 18.470 | 24.766 | 0.099 | 18.785 |
| elasticsearch | concurrent | 50/0 | 6.813 | 11.100 | 11.535 | 11.244 | 14.139 | 0.127 | 11.715 |

milvus: mean search speedup **1.68x**.
Identical final fused result ID order: 50/50 queries (last observation).

elasticsearch: mean search speedup **1.62x**.
Identical final fused result ID order: 50/50 queries (last observation).

Quick baseline on a shared host, not a statistically robust capacity test. Modes run in the recorded order, so caching and background load may affect the difference. Use --modes concurrent sequential for an order-reversed follow-up.
