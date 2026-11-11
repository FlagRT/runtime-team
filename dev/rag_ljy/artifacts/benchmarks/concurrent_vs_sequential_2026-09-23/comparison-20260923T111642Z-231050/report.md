# Retrieval comparison

Run started: **2026-09-23 19:16:42 Asia/Shanghai (11:16:42 UTC)**.

Original run ID: `comparison-20260923T111642Z-231050`. Copied from the RAG container; no measurements rerun or modified.

50 identical queries; 1 measured pass(es) per mode; 3 excluded warm-ups per mode. Embeddings computed once; no reranking.

Each database runs separately. Requests are serial; concurrent means BM25 and dense overlap within one request. All times are client-observed milliseconds.

| Database | Mode | Success/fail | BM25 mean | Dense mean | Search mean | Search P50 | Search P95 | RRF mean | Total mean |
|---|---|---|---|---|---|---|---|---|---|
| milvus | sequential | 50/0 | 3.750 | 4.180 | 7.940 | 7.808 | 9.273 | 0.099 | 8.070 |
| milvus | concurrent | 50/0 | 3.627 | 4.577 | 4.954 | 4.481 | 5.539 | 0.105 | 5.099 |
| elasticsearch | sequential | 50/0 | 10.192 | 21.827 | 32.028 | 29.361 | 47.240 | 0.098 | 32.158 |
| elasticsearch | concurrent | 50/0 | 8.769 | 15.689 | 16.268 | 15.931 | 18.420 | 0.239 | 16.686 |

milvus: mean search speedup **1.60x**.
Identical final fused result ID order: 50/50 queries (last observation).
elasticsearch: mean search speedup **1.97x**.
Identical final fused result ID order: 50/50 queries (last observation).

Verification: all four runs used the identical 50 queries, query order, and precomputed vectors; 200/200 measured retrievals succeeded.

Result equality means the ordered final fused document IDs matched between sequential and concurrent modes within each database (50/50 each). It does not assert equality between Milvus and Elasticsearch, score equality, or identical timing on future runs.

Artifacts: `queries.jsonl` and `query_vectors.json` freeze the inputs; `result_ids.json` records final ranked IDs; each database/mode directory contains `raw.csv`, `summary.json`, and its query list.

Quick baseline on a shared host, not a statistically robust capacity test. Modes run in the recorded order, so caching and background load may affect the difference. Use --modes concurrent sequential for an order-reversed follow-up.
