# Retrieval benchmark results

- [2026-09-23: same 50 NFCorpus queries, sequential vs concurrent](2026-09-23/comparison-20260923T111642Z-231050/report.md). Milvus and Elasticsearch measured separately; all 200 retrievals succeeded. Raw timings, frozen inputs, ranked result IDs, and checksums are retained alongside the report.

To rerun inside `rag-ljy-vllm-910c` with exactly the saved 50 query texts:

```bash
cd /workspace/dev/rag_ljy
source .venv/bin/activate
export DC_BACKEND=ascend
python scripts/benchmark_retrieval_comparison.py \
  --queries benchmarks/2026-09-23/comparison-20260923T111642Z-231050/queries.jsonl \
  --device npu:0 --limit 50 --warmup 3 --repeats 1 \
  --output-dir benchmarks/2026-09-23
```

Change the output date directory for a later run. Each invocation creates a unique run directory and timestamps its report. The script computes vectors once per invocation and reuses them for all modes/databases. `--backends milvus` or `--backends elasticsearch` selects a single database; the default runs both separately. `--modes concurrent sequential` reverses mode order to investigate cache/order effects. This measures retrieval and RRF only, excluding embedding and reranking.
