# device-context compatibility check — 2026-10-28

Checkout HEAD: `ca76d64befc04d3a8050f7a33d22cb177c20c8fd`. RAG has uncommitted changes; this is a working-tree check.

Result: the current RAG initialization and memory-probe calls are compatible with the current checkout runtime in offline checks. Actual vendor bindings and NPU operation were not tested.

## Verification

- Device-context Ascend offline self-check: **92 passed, 0 failed, 2 skipped** (stub capability boundaries).
- RAG integration checks: **7 passed**, using actual RAG/runtime/Ascend backend code with simulated Torch/NPU primitives.

- Actual checkout runtime imports and all six RAG API signatures match
- RAG device initialization routes through the actual Ascend backend
- RAG accepts current runtime memory fields and MiB values
- Repeated sampling observes changes through runtime.memory_stats
- Invisible device is rejected before binding
- Native telemetry failure is reported unavailable
- Sampling recovers after telemetry becomes available

The RAG container was stopped (Exited 137). No container or model service was started. No change to the RAG integration is needed on the basis of these checks.

Memory scope remains backend_reported_unverified; the compatibility check does not establish device-wide accuracy or OOM-prevention effectiveness.

## Source hashes

- `dev/rag_ljy/src/rag_engine/device.py`: `a4c714c93a0352fd5b47cc33c4291114f9c7ccf7882ba3564779c3f452334050`
- `dev/rag_ljy/src/rag_engine/memory_probe.py`: `835f6f2ed2058843649be87377aeb15b8ea6c9df77dd56812801aabf42044959`
- `dev/device-context/prototype/runtime/__init__.py`: `a3bef5b9b644d89703366371d3ef3eedd9223a59c5432dfd3ac39a30cda21673`
- `dev/device-context/prototype/runtime/backends/registry.py`: `f6ca4347d0a7c41bb53828e963f8ebba947195c5bbcd929158d53ee50eeedf44`
- `dev/device-context/prototype/runtime/backends/ascend/backend.py`: `4393ed829994f352f5e4e4b925569aaa54a68205f1045c6b8609b93dd00a434c`
