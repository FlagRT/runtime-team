# Device-context integration study

Reviewed 2026-09-22 against checkout `8575cab` and the supplied service-startup
standard v1.1. This records the original study. The first-stage direct integration
has since been implemented in `src/rag_engine/device.py`; see the README's
device-context section for current commands. The proposed code and migration table
below describe the study, not a list of changes all implemented. Error recovery,
service migration, and cross-chip hardware acceptance remain future work.

## Recommendation

Integrate the in-process `device-context/prototype/runtime` API first. Keep the
existing Transformers embedding and reranking implementations. The smallest
integration boundary is `src/rag_engine/npu.py`, which already centralizes device
initialization for both models.

The runtime abstracts device selection, memory statistics, streams/events, error
classification, and recovery. It does not implement model operators or supply a
portable PyTorch installation. Each accelerator still needs its matching vendor
PyTorch, drivers, container configuration, and supported model operations.

## What exists in this checkout

Sources are relative to `dev/`:

| Component | Evidence | Implication |
|---|---|---|
| Public runtime API | `device-context/prototype/runtime/__init__.py` | `use`, `set_device`, `memory_stats`, `translate_error`, and recovery are available. |
| Backend interface | `device-context/prototype/runtime/backends/base.py` | `device_type` supplies the PyTorch device prefix; operator dispatch remains with vendor PyTorch. |
| Ascend backend | `device-context/prototype/runtime/backends/ascend/backend.py` | `ascend` resolves to `npu`; the backend imports `torch_npu` internally. |
| Kunlun backend | `device-context/prototype/runtime/backends/kunlun/backend.py` | `kunlun` resolves to `cuda` through XPytorch, not `xpu`. |
| FlagOS backend | `device-context/prototype/runtime/backends/flagos/backend.py` | Separate `flagos` device namespace and `torch_fl` stack; not a universal selector for all vendors. |
| Service launcher | Supplied guide: `prototype/scripts/serve_standard.sh` | Absent from this checkout. Obtain the matching device-context revision before integrating it. |
| Cambricon | Mentioned in supplied guide | No Cambricon Python backend in this checkout; the guide also marks its launcher branch as unverified on hardware. |

A read-only import check confirmed that the local module resolves correctly and
registers `ascend`, `flagos`, and `kunlun`. Registration is lazy and does **not**
prove that hardware or vendor dependencies are available. No accelerator inference
or hardware conformance was run for this study.

## Proposed direct integration

Expose a backend and logical device ordinal separately, for example
`--backend ascend --device-index 0`. Backend names identify the vendor stack;
device strings identify the PyTorch namespace. Do not infer Kunlun from `cuda`,
because that namespace can also identify NVIDIA in another environment.

Illustrative initialization using the existing API:

```python
def prepare_runtime_device(backend_name: str, ordinal: int):
    import runtime

    backend = runtime.use(backend_name)
    count = runtime.device_count()
    if not 0 <= ordinal < count:
        raise ValueError(f"Device index {ordinal} outside visible range 0..{count - 1}")
    runtime.set_device(ordinal)  # Loads the selected vendor implementation.

    import torch
    return torch, torch.device(f"{backend.device_type}:{ordinal}")
```

This is a design example, not a helper already installed in RAG. For this source
checkout, the parent directory of `runtime` must be importable. Inside the current
container, where the repository is mounted at `/workspace`, that directory is
`/workspace/dev/device-context/prototype`. Add it to the launch environment's
`PYTHONPATH`, preserving existing entries. The prototype has no packaging manifest
for a straightforward editable installation. Prefer an upstream packaged release
when available rather than copying runtime sources into RAG.

| RAG area | Proposed change |
|---|---|
| `src/rag_engine/npu.py` | Introduce a neutral adapter such as `device.py`; delegate accelerator setup to runtime. Retain an explicit CPU path because runtime has no CPU backend here. Preserve legacy device arguments through a compatibility layer if needed. |
| `embedding.py`, `reranker.py` | Use the adapter; retain `.to(resolved_device)` for models and inputs. Keep pooling, prompts, truncation, normalization, and scoring unchanged during the first migration. |
| Dtype selection | Make dtype explicit/configurable. Current logic chooses BF16 only for `npu`/`cuda`; it would silently choose FP32 for `flagos` or a future `mlu` device. The runtime has no dtype capability API. Validate BF16 on the target model/backend. |
| `query_runtime.py`, `server.py` | Initialize one backend/device per process and pass that choice to both models. Preserve model initialization and inference on the same dedicated executor thread. |
| Query, ingestion, benchmark CLIs | Propagate the same backend/index settings to every model construction path, including direct embedder construction outside `create_pipeline`. Record backend, resolved device, dtype, and vendor stack in benchmark metadata. |
| `scripts/checks/check_npu.py` | Add a generic runtime check: initialize, count, probe, report memory, allocate a small tensor on the resolved device, and verify its result. |
| `scripts/runtime/setup_npu_env.sh` | Generalize vendor dependency checks and constraints without replacing the image's PyTorch stack. It currently unconditionally imports and requires `torch_npu`. |
| Compose/deployment | Use each vendor's approved environment. The current Ascend image and inherited `/dev/davinci*`/driver mounts are not portable to P800. |

The current runtime registry stores the selected backend globally, so do not switch
backends per HTTP request. The RAG server already keeps accelerator work on one
thread; concurrent BM25/dense searches only make database calls and do not require
device streams. There is no reason to introduce multi-stream execution for this
initial migration. If it is added later, follow the interface contract's
`record_stream` lifetime discipline.

Preserve CPU behavior explicitly. The current RAG helper also accepts CUDA; there
is no dedicated NVIDIA backend in this runtime checkout. Decide whether to keep
that native path or require an upstream NVIDIA backend before replacing it.

## Error handling and recovery

Translate original accelerator exceptions with
`runtime.translate_error(exc, location="rag.embedding")` (and an equivalent
reranker location), preserving exception chaining and the full original message.
Keep database/validation failures outside this accelerator error boundary.

Use the resulting `disposition` instead of matching vendor error strings. Start
with reporting and bounded failure handling; retry/replay needs a deliberate
budget and safe operation boundaries. Fatal recovery must account for loaded
models and tensors; merely resetting a device does not reload the model pair.
Kunlun does not support real device recovery here. Its bounded synchronization
also has narrower guarantees than a universal cancellation mechanism. Consult
`supports()` and backend implementation details before adding recovery or timeout
policies. A timeout does not establish that outstanding device work has stopped.

## How the supplied service standard fits

The guide describes a second integration route:

```text
RAG embedding client -> HTTP /v1/embeddings -> standard launcher -> vendor vLLM
```

After obtaining the missing launcher, reuse it as the sole vLLM startup entry
point instead of duplicating its plugin, environment, readiness, smoke, and cleanup
logic. Its `DC_BACKEND` setting belongs to that launcher; the Python runtime API
does not automatically consume it. RAG must explicitly read configuration and
call `runtime.use(...)` for an in-process integration.

An HTTP embedder could implement the same `encode_queries`/`encode_documents`
interface. It must preserve query-only instructions, response order, output
dimension, pooling/normalization semantics, and truncation policy. The current
local models use an 8192-token limit; the standard service defaults to 4096.
Equal output dimension alone does not establish compatible embeddings. Compare
vectors and retrieval behavior before querying an existing index with a new
embedding implementation; re-embed if compatibility is not established.

The supplied Ascend example serves a generation model, while the P800 example
serves an embedding model. They are not interchangeable RAG configurations.
Confirm that the upstream launcher exposes the embedding task on Ascend as well;
the unavailable script prevents verifying how task selection is implemented.

The reranker needs separate work: it calculates
`softmax([logit_no, logit_yes])[yes]` at the final prompt position using its exact
prefix/suffix. Neither an embedding response nor generated yes/no text preserves
that score. The supplied standard does not establish an equivalent scoring
endpoint. Keep reranking local initially, or verify a service scoring contract
against the existing implementation before replacing it.

A service split also avoids requiring the RAG client and accelerator server to
share a Python interpreter. This matters because RAG requires Python >=3.11,
whereas the documented P800 environment is `python310_torch29_cuda`. Direct
integration requires resolving that mismatch with a supported vendor environment
or a separately tested Python compatibility change; do not simply lower the
package requirement without validating dependencies and tests.

## Suggested acceptance sequence

1. Pin the device-context revision and vendor environments; obtain the missing
   standard launcher if pursuing the service route.
2. Integrate the adapter on existing 910C hardware first, with the same model
   revisions, dtype, device, prompts, and token limits. Run existing RAG tests and
   runtime conformance; verify actual device placement and both model outputs.
3. Compare embeddings, reranker scores/ranks, and retrieval results against the
   current implementation. Include ingestion, repeated HTTP queries, and concurrent
   database retrieval. Set numerical tolerances rather than requiring bit equality.
4. Repeat on P800 after resolving Python/vendor setup. Confirm model operator and
   dtype support through real forwards; device API conformance alone is insufficient.
5. Measure latency/memory separately from correctness. Test error propagation and
   service readiness under failure before adding retries or recovery.

The direct API route is a small application refactor plus vendor deployment and
validation work. The service route is a broader model-provider migration and is
most immediately applicable to embedding and future generation.
