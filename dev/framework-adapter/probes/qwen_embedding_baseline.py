"""Observe real Transformers Qwen3 inputs and replay native representative ops.

Diagnostic eager path, not vLLM/compile/performance coverage. No registration,
kernel replacement, automatic device fallback, network download, or training.
Only metadata/results are serialized; weights and activations are not saved.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import sys
import traceback


def last_token_indices(mask_rows):
    """Works for left and right padding; reject all-padding rows."""
    indices = []
    for row in mask_rows:
        active = [i for i, value in enumerate(row) if value]
        if not active:
            raise ValueError("Cannot pool an all-padding row")
        indices.append(active[-1])
    return indices


def tolerance(dtype_name):
    return {"float32": 2e-5, "float16": 0.005, "bfloat16": 0.03}[dtype_name]


def source_identity(callable_):
    fn = getattr(callable_, "__func__", callable_)
    path = inspect.getsourcefile(fn)
    return {"file": path, "function": getattr(fn, "__qualname__", str(fn)),
            "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None}


def initialize_device(torch, device_spec, runtime_root, result):
    # Runtime.use selects a lazy backend; device_count triggers vendor import.
    # Register PrivateUse1 BEFORE parsing npu (especially with AUTOLOAD=0).
    kind = device_spec.split(":", 1)[0]
    if kind == "npu":
        if not runtime_root:
            raise ValueError("NPU tests require --runtime-root (device-context/prototype)")
        sys.path.insert(0, str(Path(runtime_root).resolve()))
        import runtime
        runtime.use("ascend")
        result["runtime"] = {"version": runtime.__version__, "file": runtime.__file__}
        result["stage"] = "device_initialization"
        result["visible_devices"] = runtime.device_count()
        ordinal = int(device_spec.split(":", 1)[1]) if ":" in device_spec else 0
        runtime.set_device(ordinal)
        device = torch.device(device_spec)
        torch.ones(1, device=device).cpu()  # fail before model work if device is unusable
    elif kind == "cpu":
        device = torch.device(device_spec)
    else:
        raise ValueError("Only explicit cpu or npu devices are supported")
    return device


def run(args, result):
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    device = initialize_device(torch, args.device, args.runtime_root, result)
    result["stage"] = "model_and_operator_baseline"

    config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3":
        raise ValueError(f"Expected Qwen3 model, got {config.model_type}")
    result["config"] = {k: getattr(config, k, None) for k in (
        "model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim", "rms_norm_eps")}
    result["config_sha256"] = hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    # Eager is an explicitly labeled diagnostic setting, not the serving default.
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                     dtype=getattr(torch, args.dtype), attn_implementation="eager")
    model = model.to(device).eval()
    result["attention_implementation"] = model.config._attn_implementation
    counts, signatures, pending, handles = Counter(), set(), {}, []
    cases = result["operator_cases"]
    tol = tolerance(args.dtype)
    sources = {}

    def classify(module):
        name = type(module).__name__
        if name == "Qwen3RMSNorm":
            return "rms_norm"
        if name in ("SiLUActivation", "SiLU"):
            return "silu"
        return None

    def pre(module, inputs, kwargs):
        family = classify(module)
        qualified = type(module).__module__ + "." + type(module).__name__
        counts[qualified] += 1
        x = inputs[0] if inputs else next(v for v in kwargs.values() if isinstance(v, torch.Tensor))
        eps = getattr(module, "variance_epsilon", None)
        key = (family, tuple(x.shape), tuple(x.stride()), str(x.dtype), eps)
        if key in signatures:
            return
        if len(signatures) >= 32:
            raise RuntimeError("Diagnostic sample cap reached; reduce the workload")
        signatures.add(key)
        sources.setdefault(qualified, source_identity(module.forward))
        weight = getattr(module, "weight", None)
        pending[id(module)] = {
            "x": x, "input_cpu": x.detach().cpu().clone(),
            "weight_cpu": weight.detach().cpu().clone() if weight is not None else None,
            "case": {"op": family, "module": module._framework_probe_name,
                     "class": qualified, "shape": list(x.shape), "stride": list(x.stride()),
                     "dtype": str(x.dtype), "device": str(x.device), "eps": eps,
                     "atol": tol, "rtol": tol, "status": "started"},
        }

    def post(module, inputs, kwargs, output):
        sample = pending.pop(id(module), None)
        if sample is None:
            return
        case = sample["case"]
        cases.append(case)
        x, cpu, weight = sample["x"], sample["input_cpu"], sample["weight_cpu"]
        ref_input = cpu.float()
        if case["op"] == "silu":
            reference = F.silu(ref_input)
        else:
            reference = ref_input * torch.rsqrt(ref_input.square().mean(-1, keepdim=True) + case["eps"])
            reference = reference * weight.float()
        actual = output.detach().cpu().float()
        torch.testing.assert_close(actual, reference, atol=tol, rtol=tol)
        torch.testing.assert_close(x.detach().cpu(), cpu, atol=0, rtol=0)
        if weight is not None:
            torch.testing.assert_close(module.weight.detach().cpu(), weight, atol=0, rtol=0)
        if output.device != x.device or output.dtype != x.dtype:
            raise AssertionError("Output device or dtype changed")
        # Direct forward bypasses Module hooks, replaying the same native module.
        replay_input = x.detach().clone(memory_format=torch.preserve_format)
        replay = module.forward(replay_input)
        torch.testing.assert_close(replay.detach().cpu().float(), actual, atol=tol, rtol=tol)
        torch.testing.assert_close(replay_input.cpu(), cpu, atol=0, rtol=0)
        if weight is not None:
            torch.testing.assert_close(module.weight.detach().cpu(), weight, atol=0, rtol=0)
        case.update(status="pass", max_abs_error=float((actual-reference).abs().max()),
                    replay_matches=True, input_unchanged=True, weight_unchanged=True,
                    output_shape=list(output.shape))
        print(json.dumps({"stage": "operator", **case}, ensure_ascii=False), flush=True)

    for name, module in model.named_modules():
        if classify(module):
            module._framework_probe_name = name
            handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(module.register_forward_hook(post, with_kwargs=True))
    if not handles:
        raise RuntimeError("No expected model modules found; do not claim operator coverage")

    batches = [["如何申请退款"], ["如何申请退款", "退款流程是怎样的", "今天天气怎么样"]]
    try:
        with torch.inference_mode():
            for texts in batches:
                encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                indices = last_token_indices(encoded["attention_mask"].tolist())
                encoded = encoded.to(device)
                hidden = model(**encoded, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(texts), device=device), torch.tensor(indices, device=device)]
                vectors = F.normalize(pooled.float(), p=2, dim=-1).cpu()
                if vectors.shape != (len(texts), config.hidden_size) or not bool(torch.isfinite(vectors).all()):
                    raise AssertionError("Invalid embedding shape or values")
                torch.testing.assert_close(vectors.norm(dim=-1), torch.ones(len(texts)), atol=1e-5, rtol=1e-5)
                result["model_batches"].append({"batch": len(texts), "token_shape": list(encoded["input_ids"].shape),
                    "pool_indices": indices, "output_shape": list(vectors.shape), "finite": True, "unit_norm": True})
                print(json.dumps({"stage": "model", **result["model_batches"][-1]}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        result["module_call_counts"] = dict(counts)
        result["sources"] = sources
        for sample in pending.values():
            sample["case"]["status"] = "incomplete"
            cases.append(sample["case"])
    if {case["op"] for case in cases} != {"silu", "rms_norm"}:
        raise AssertionError("Both target operator families were not observed")
    result["status"] = "pass"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True, help="Explicit cpu or npu:0; never automatically falls back")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        parser.error("Output already exists; choose a new file to preserve evidence")
    result = {"status": "started", "framework": "transformers", "mode": "diagnostic-eager",
              "device": args.device, "dtype": args.dtype, "model": args.model,
              "operator_cases": [], "model_batches": [],
              "scope": "native model-derived ops, no fallback or serving/performance claim"}
    try:
        run(args, result)
    except Exception as exc:
        result.update(status="fail", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
