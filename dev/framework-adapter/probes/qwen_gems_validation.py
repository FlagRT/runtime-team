"""NPU Qwen3 scoped replacement validation, not serving/performance acceptance."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import traceback

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.activations import SiLUActivation
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from qwen_embedding_baseline import initialize_device, last_token_indices, tolerance
from qwen_scoped_adapter import QwenScopedGems


def assert_routes(scope, op, backend, reason=None, stage=None):
    matches = [v for k, v in scope.counts.items()
               if k[0] == op and k[1] == backend and (reason is None or k[2] == reason)
               and (stage is None or k[3] == stage)]
    if not matches or sum(matches) == 0:
        raise AssertionError(f"Missing expected route: {op}/{backend}/{reason}/{stage}")


def run(args, result):
    device = initialize_device(torch, "npu:0", args.runtime_root, result)
    dtype = getattr(torch, args.dtype)
    tol = tolerance(args.dtype)
    result["packages"] = {n: importlib.metadata.version(n) for n in (
        "torch", "torch-npu", "triton", "triton-ascend", "transformers")}
    import flag_gems
    result["gems_source"] = flag_gems.__file__
    result["stage"] = "operator_routes"
    # Representative model shapes established by the 0916 probe.
    specs = [("rms_norm", (1, 4, 1024)), ("rms_norm", (1, 4, 16, 128)),
             ("rms_norm", (1, 4, 8, 128)), ("silu", (1, 4, 3072)),
             ("rms_norm", (3, 6, 1024)), ("rms_norm", (3, 6, 16, 128)),
             ("rms_norm", (3, 6, 8, 128)), ("silu", (3, 6, 3072))]
    torch.manual_seed(22)
    with torch.inference_mode():
        for op, shape in specs:
            module = (Qwen3RMSNorm(shape[-1], eps=1e-6) if op == "rms_norm"
                      else SiLUActivation()).to(device=device, dtype=dtype).eval()
            x = torch.randn(shape, dtype=dtype, device=device)
            before = x.cpu().clone()
            if op == "rms_norm":
                module.weight.copy_(torch.randn(shape[-1], device=device, dtype=dtype))
            saved = {n: p.cpu().clone() for n, p in module.named_parameters()}
            native = module(x).cpu()
            with QwenScopedGems(module, allowed_ops=(op,)) as scope:
                actual = module(x).cpu()
            assert_routes(scope, op, "flaggems", stage="completed")
            torch.testing.assert_close(actual, native, atol=tol, rtol=tol)
            torch.testing.assert_close(x.cpu(), before, atol=0, rtol=0)
            for n, p in module.named_parameters():
                torch.testing.assert_close(p.cpu(), saved[n], atol=0, rtol=0)
            assert "forward" not in module.__dict__
            torch.testing.assert_close(module(x).cpu(), native, atol=0, rtol=0)
            result["operator_cases"].append(dict(op=op, shape=shape, status="pass",
                max_abs_error=float((actual.float()-native.float()).abs().max()),
                input_weight_unchanged=True, restored=True, routes=scope.summary()))

        # Real NPU preflight fallback, not an injected hardware failure.
        for reason in ("disabled", "operator_not_enabled", "dtype", "layout"):
            module = SiLUActivation().to(device).eval()
            x = torch.randn(2, 8, device=device, dtype=torch.float32 if reason == "dtype" else dtype)
            if reason == "layout":
                x = x.t()
            native = module(x).cpu()
            with QwenScopedGems(module, enabled=reason != "disabled",
                                allowed_ops=() if reason == "operator_not_enabled" else ("silu",)) as scope:
                actual = module(x).cpu()
            assert_routes(scope, "silu", "native", reason)
            assert not any(k[1] == "flaggems" for k in scope.counts)
            torch.testing.assert_close(actual, native, atol=0, rtol=0)
            result["fallback_cases"].append(dict(reason=reason, status="pass", routes=scope.summary()))

    result["stage"] = "model_routes"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                     dtype=dtype, attn_implementation="eager").to(device).eval()
    if model.config.model_type != "qwen3":
        raise ValueError("Only verified Qwen3 model family is permitted")
    result["config_sha256"] = hashlib.sha256((Path(args.model)/"config.json").read_bytes()).hexdigest()
    targets = [m for m in model.modules() if type(m) in (Qwen3RMSNorm, SiLUActivation)]
    batches = [["如何申请退款"], ["如何申请退款", "退款流程是怎样的", "今天天气怎么样"]]
    with torch.inference_mode():
        for texts in batches:
            encoded = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            indices = last_token_indices(encoded["attention_mask"].cpu().tolist())
            def forward():
                hidden = model(**encoded, use_cache=False).last_hidden_state
                pooled = hidden[torch.arange(len(texts), device=device), torch.tensor(indices, device=device)]
                vector = torch.nn.functional.normalize(pooled.float(), dim=-1)
                torch.npu.synchronize(device)
                return hidden.cpu(), vector.cpu()
            native_h, native_v = forward()
            modes = [("disabled", False, ("silu",)), ("silu_only", True, ("silu",))]
            if args.allow_rms_experiment:
                modes += [("enabled", True, ("silu", "rms_norm")),
                          ("rms_only", True, ("rms_norm",))]
            for label, enabled, ops in modes:
                with QwenScopedGems(model, enabled=enabled, allowed_ops=ops) as scope:
                    actual_h, actual_v = forward()
                assert all("forward" not in m.__dict__ for m in targets)
                case = dict(batch=len(texts), mode=label, routes=scope.summary(), status="started",
                            hidden_max_abs_error=float((actual_h.float()-native_h.float()).abs().max()),
                            embedding_max_abs_error=float((actual_v-native_v).abs().max()))
                result["model_cases"].append(case)
                if enabled:
                    for op in ("silu", "rms_norm"):
                        assert_routes(scope, op, "flaggems" if op in ops else "native",
                                      stage="completed" if op in ops else "selected")
                else:
                    assert not any(k[1] == "flaggems" for k in scope.counts)
                case["checks"] = {}
                # Numerical comparison failures are evidence, not execution errors.
                # Continue independent A/B modes without relaxing tolerances.
                for name, actual, expected in (("hidden", actual_h, native_h),
                                               ("embedding", actual_v, native_v)):
                    try:
                        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                        case["checks"][name] = {"status": "pass"}
                    except AssertionError as exc:
                        case["checks"][name] = {"status": "fail", "error": str(exc)}
                case["status"] = ("pass" if all(c["status"] == "pass" for c in case["checks"].values())
                                  else "fail")
                print(json.dumps({k:v for k,v in case.items() if k != "routes"}), flush=True)
            restored_h, restored_v = forward()
            torch.testing.assert_close(restored_h, native_h, atol=0, rtol=0)
            torch.testing.assert_close(restored_v, native_v, atol=0, rtol=0)
    result["status"] = "pass" if all(c["status"] == "pass" for c in result["model_cases"]) else "fail"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--out", required=True)
    p.add_argument("--allow-rms-experiment", action="store_true",
                   help="Opt into RMSNorm model replacement; known FP16 hidden-state mismatch, NOT validated")
    args = p.parse_args()
    path = Path(args.out)
    # Reserve output exclusively, preserving failed and partial evidence too.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        result = dict(status="started", dtype=args.dtype, device="npu:0", operator_cases=[],
                      fallback_cases=[], model_cases=[], mode="diagnostic-eager-synchronized",
                      rms_model_replacement="experimental" if args.allow_rms_experiment else "disabled",
                      scope="single-thread inference; no vLLM, training, cross-chip replay or performance claim")
        try:
            run(args, result)
        except Exception as exc:
            result.update(status="fail", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
        finally:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
