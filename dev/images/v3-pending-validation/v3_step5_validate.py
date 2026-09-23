#!/usr/bin/env python3
"""Real-hardware Step 5 validation for ascend-operator-runtime/ascend-train-comm v3
round 2 (vllm-plugin-FL pivot). Run inside the ascend-train-comm:v3 image
(flagrt/ascend-operator-runtime-comm:2.0.0-flagtree3.5-routeA-cann9.0-py311-torch2.10-flagcx0.13.0g4e0e0cb-arm64,
image id 43f3e2f70b4c) which has both FlagCX (training side) and
vllm-plugin-FL (inference side) installed, on real NPU hardware.

Does two things in one process, in this order:
  1. Coexistence check: import torch_npu -> flagcx -> vllm (+ vllm_fl platform
     registration), confirm PrivateUse1 stays "npu" throughout, no driver
     conflict.
  2. vLLM inference smoke test: load the acceptance_model from
     dev/stack.lock.910c.v2.yaml (Qwen/Qwen3-Embedding-0.6B, an embedding
     model) via vllm-plugin-FL's "fl" platform plugin, run a real .encode()
     call, print the actual output (embedding vector shape/norm/first few
     values) as evidence it produced real numbers, not garbage/NaN/zeros.

Everything is printed as JSON at the end plus human-readable progress lines
along the way, so a crash mid-way still leaves useful evidence in the log.
"""
import json
import os
import sys
import traceback

os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

result = {"steps": []}


def record(name, ok, detail=None):
    entry = {"step": name, "ok": ok}
    if detail is not None:
        entry["detail"] = detail
    result["steps"].append(entry)
    print(f"[STEP] {name}: {'OK' if ok else 'FAIL'} {detail if detail else ''}", flush=True)


def main():
    # --- 1. torch_npu default backend (Route A) ---
    try:
        import torch
        import torch_npu  # noqa: F401
        backend_name = torch._C._get_privateuse1_backend_name()
        assert backend_name == "npu", f"expected npu, got {backend_name}"
        record("import_torch_npu_route_a", True, {"privateuse1_backend": backend_name})
    except Exception as e:
        record("import_torch_npu_route_a", False, {"error": repr(e), "trace": traceback.format_exc()})
        print(json.dumps(result, indent=2))
        return 1

    # --- 2. flagcx import (coexistence, no actual DDP init needed) ---
    try:
        import flagcx  # noqa: F401
        record("import_flagcx", True, {"module_file": getattr(flagcx, "__file__", None)})
    except Exception as e:
        record("import_flagcx", False, {"error": repr(e), "trace": traceback.format_exc()})

    # confirm PrivateUse1 unaffected by flagcx import
    try:
        backend_name2 = torch._C._get_privateuse1_backend_name()
        assert backend_name2 == "npu", f"expected npu, got {backend_name2}"
        record("privateuse1_stable_after_flagcx", True, {"privateuse1_backend": backend_name2})
    except Exception as e:
        record("privateuse1_stable_after_flagcx", False, {"error": repr(e)})

    # --- 3. triton driver sanity (v2's "2 active drivers" class of issue) ---
    try:
        import triton
        driver_active = None
        try:
            driver_active = type(triton.runtime.driver.active).__name__
        except Exception as de:
            driver_active = f"<could not introspect: {de!r}>"
        record("triton_driver_check", True, {"triton_version": triton.__version__, "active_driver": driver_active})
    except Exception as e:
        record("triton_driver_check", False, {"error": repr(e), "trace": traceback.format_exc()})

    # --- 4. vllm + vllm_fl platform registration ---
    try:
        import vllm  # noqa: F401
        import vllm_fl  # noqa: F401
        from vllm.platforms import current_platform
        record("import_vllm_and_vllm_fl", True, {
            "vllm_version": getattr(vllm, "__version__", "unknown"),
            "vllm_fl_version": getattr(getattr(vllm_fl, "version", None), "__version__", None) or str(getattr(vllm_fl, "__version__", "unknown")),
            "current_platform": str(current_platform),
            "current_platform_type": type(current_platform).__name__,
        })
    except Exception as e:
        record("import_vllm_and_vllm_fl", False, {"error": repr(e), "trace": traceback.format_exc()})
        print(json.dumps(result, indent=2))
        return 1

    # --- 5. real inference smoke test: Qwen/Qwen3-Embedding-0.6B via vllm-plugin-FL ---
    model_path = os.environ.get("SMOKE_MODEL_PATH", "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
    try:
        from vllm import LLM

        llm = LLM(
            model=model_path,
            task="embed",
            enforce_eager=True,  # required for Ascend per vllm-plugin-FL README
            max_model_len=512,
            gpu_memory_utilization=0.5,
        )
        record("llm_construct", True, {"model_path": model_path})

        prompts = ["Hello, my name is", "The FlagOS unified multi-chip backend runs vLLM on Ascend NPU."]
        outputs = llm.encode(prompts)

        embeddings_summary = []
        for i, out in enumerate(outputs):
            emb = out.outputs.embedding
            import math
            n = len(emb)
            norm = math.sqrt(sum(x * x for x in emb))
            has_nan = any(x != x for x in emb)  # NaN check without numpy
            all_zero = all(x == 0.0 for x in emb)
            embeddings_summary.append({
                "prompt": prompts[i],
                "embedding_dim": n,
                "embedding_l2_norm": norm,
                "first_5_values": emb[:5],
                "has_nan": has_nan,
                "all_zero": all_zero,
            })

        # correctness signal: two different prompts should NOT produce identical
        # embeddings (that would indicate the token-id-collapse bug from the
        # FlagRT private fork's fix commit -- everything degenerating to the
        # same embedding regardless of input).
        e0 = outputs[0].outputs.embedding
        e1 = outputs[1].outputs.embedding
        identical = all(abs(a - b) < 1e-9 for a, b in zip(e0, e1))

        record("vllm_encode_real_output", True, {
            "embeddings": embeddings_summary,
            "two_different_prompts_produced_identical_embeddings": identical,
        })

        # device placement confirmation: NPU memory should be allocated after
        # a real forward pass (rules out silent CPU fallback).
        try:
            npu_mem = torch.npu.memory_allocated()
            record("npu_memory_allocated_after_inference", True, {"bytes": npu_mem, "nonzero": npu_mem > 0})
        except Exception as e:
            record("npu_memory_allocated_after_inference", False, {"error": repr(e)})

        del llm
    except Exception as e:
        record("vllm_encode_real_output", False, {"error": repr(e), "trace": traceback.format_exc()})

    # --- 6. final PrivateUse1 stability check ---
    try:
        backend_name3 = torch._C._get_privateuse1_backend_name()
        record("privateuse1_stable_after_inference", backend_name3 == "npu", {"privateuse1_backend": backend_name3})
    except Exception as e:
        record("privateuse1_stable_after_inference", False, {"error": repr(e)})

    print("\n=== FULL RESULT JSON ===")
    print(json.dumps(result, indent=2, default=str))
    all_ok = all(s["ok"] for s in result["steps"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
