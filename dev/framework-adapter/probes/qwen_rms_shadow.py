"""Same-input RMSNorm diagnostics. Hooks observe, never replace model outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

import torch
from transformers import AutoModel, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from qwen_embedding_baseline import initialize_device, tolerance


def compare(actual, expected, tol):
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff = (a - b).abs()
    return dict(finite=finite, elements=a.numel(),
                different=int((a != b).sum()),
                outside_tolerance=int((diff > tol + tol * b.abs()).sum()),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()))


def run(args, result):
    device = initialize_device(torch, "npu:0", args.runtime_root, result)
    import flag_gems
    result["gems_source"] = flag_gems.__file__
    dtype = getattr(torch, args.dtype)
    tol = tolerance(args.dtype)
    result["tolerance"] = dict(atol=tol, rtol=tol)
    result["config_sha256"] = hashlib.sha256((Path(args.model) / "config.json").read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
                                     dtype=dtype, attn_implementation="eager").to(device).eval()
    if model.config.model_type != "qwen3":
        raise ValueError("Only Qwen3 is supported")
    batches = [["如何申请退款"], ["如何申请退款", "退款流程是怎样的", "今天天气怎么样"]]
    with torch.inference_mode():
        for texts in batches:
            encoded = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            native = model(**encoded, use_cache=False).last_hidden_state.cpu()
            batch = dict(batch=len(texts), calls=[], output_unchanged=False)
            result["batches"].append(batch)

            def make_hook(name):
                def hook(module, inputs, output):
                    x = inputs[0]
                    xc, wc, yc = x.cpu().clone(), module.weight.cpu().clone(), output.cpu().clone()
                    eps = module.variance_epsilon
                    variants = {}
                    # Compare real kernel, then two diagnostic compositions. These
                    # are not proposed production replacements or performance tests.
                    variants["gems"] = flag_gems.rms_norm(x, (x.shape[-1],), module.weight, eps).cpu()
                    unit = torch.ones_like(module.weight)
                    norm = flag_gems.rms_norm(x, (x.shape[-1],), unit, eps)
                    variants["gems_unit_then_native_weight"] = (norm * module.weight).cpu()
                    norm32 = flag_gems.rms_norm(x.float(), (x.shape[-1],), unit.float(), eps)
                    variants["gems_fp32_then_cast_weight"] = (norm32.to(x.dtype) * module.weight).cpu()
                    # Float64 diagnostic reference with native's explicit output
                    # rounding stages; not an approved model accuracy oracle.
                    xd = xc.double()
                    ref = (xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + eps)).to(xc.dtype) * wc
                    torch.npu.synchronize(device)
                    torch.testing.assert_close(x.cpu(), xc, atol=0, rtol=0)
                    torch.testing.assert_close(module.weight.cpu(), wc, atol=0, rtol=0)
                    torch.testing.assert_close(output.cpu(), yc, atol=0, rtol=0)
                    record = dict(name=name, shape=list(x.shape), stride=list(x.stride()),
                                  eps=eps, input_abs_max=float(xc.abs().max()),
                                  weight_abs_max=float(wc.abs().max()),
                                  native_vs_fp64=compare(yc, ref, tol), variants={})
                    for label, value in variants.items():
                        record["variants"][label] = dict(vs_native=compare(value, yc, tol),
                                                        vs_fp64=compare(value, ref, tol))
                    batch["calls"].append(record)
                    return None
                return hook

            handles = []
            try:
                for name, module in model.named_modules():
                    if type(module) is Qwen3RMSNorm:
                        handles.append(module.register_forward_hook(make_hook(name)))
                actual = model(**encoded, use_cache=False).last_hidden_state.cpu()
            finally:
                for handle in handles:
                    handle.remove()
            torch.testing.assert_close(actual, native, atol=0, rtol=0)
            if not batch["calls"]:
                raise AssertionError("No RMSNorm calls observed")
            batch["output_unchanged"] = True
            for label in batch["calls"][0]["variants"]:
                records = [c["variants"][label]["vs_native"] for c in batch["calls"]]
                summary = dict(batch=len(texts), variant=label, calls=len(records),
                               nonexact_calls=sum(v["different"] > 0 for v in records),
                               failing_calls=sum(not v["finite"] or v["outside_tolerance"] > 0 for v in records),
                               max_abs=max(v["max_abs"] for v in records))
                result["summary"].append(summary)
                print(json.dumps(summary), flush=True)
    # "completed" means observations collected, NOT numerical equivalence.
    result["status"] = "completed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = dict(status="started", dtype=args.dtype, batches=[], summary=[],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="same-input shadow only; model outputs not replaced; no threshold change")
    with Path(args.out).open("x") as handle:
        try:
            run(args, result)
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error=str(exc))
            traceback.print_exc()
        finally:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
