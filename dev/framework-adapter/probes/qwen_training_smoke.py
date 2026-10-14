"""Single-device native training smoke; not embedding-quality or fallback acceptance.

Uses the device team's causal-LM training formulation on the embedding checkpoint.
Any missing LM head is newly initialized. No checkpoint is saved or overwritten.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback


def run(args, result):
    import torch
    sys.path.insert(0, args.runtime_root)
    import runtime
    from transformers import AutoModelForCausalLM
    runtime.use("ascend")
    runtime.device_count()
    runtime.set_device(0)
    result["runtime_file"] = runtime.__file__
    torch.manual_seed(1234)
    model, info = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager",
        output_loading_info=True)
    result["loading_info"] = {key: sorted(value) if isinstance(value, set) else value
                              for key, value in info.items()}
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Non-finite model weights before device transfer")
    model = model.to("npu:0").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # A small, explicitly synthetic causal-LM workload, matching the team's smoke.
    ids = torch.randint(100, 30000, (1, 32), generator=torch.Generator().manual_seed(999)).to("npu:0")
    sample = model.get_input_embeddings().weight
    token = int(ids[0, 0].item())
    initial = sample[token].detach().cpu().clone()
    result["steps"] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        grad_count = sum(p.grad is not None for p in model.parameters())
        optimizer.step()
        runtime.synchronize(0)
        row = {"step": step, "loss": float(loss.item()), "grad_norm": float(norm.item()),
               "parameters_with_gradient": grad_count}
        result["steps"].append(row)
        print(json.dumps(row), flush=True)
    delta = (sample[token].detach().cpu() - initial).abs().max().item()
    result["sampled_parameter_max_update"] = delta
    if not delta > 0 or not all(s["parameters_with_gradient"] > 0 for s in result["steps"]):
        raise AssertionError("No verified parameter update/gradients")
    result["status"] = "pass"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.steps < 1 or Path(args.out).exists():
        parser.error("Require positive steps and a new output path")
    result = {"status": "fail", "scope": "native_fp32_single_device_synthetic_causal_lm",
              "operator_replacement": False, "model": args.model}
    try:
        run(args, result)
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        with open(args.out, "x") as output:
            json.dump(result, output, indent=2)
    sys.exit(0 if result["status"] == "pass" else 1)
