"""Opt-in isolated-process NPU test: real ATen SiLU forward/backward registration.

No production patch or kernel changes. Full model checks compare all gradients
and parameters, not just loss. A failed check aborts without native retry.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import logging
from pathlib import Path
import sys
import traceback


class Calls(logging.Handler):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def emit(self, record):
        message = record.getMessage()
        if message in ("GEMS SILU FORWARD", "GEMS SILU BACKWARD"):
            self.counts[message] += 1


def run(args, result):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, args.runtime_root)
    import runtime
    runtime.use("ascend")
    runtime.device_count()
    runtime.set_device(0)
    import flag_gems
    result["runtime_file"] = runtime.__file__
    result["gems_file"] = flag_gems.__file__
    calls = Calls()
    logger = logging.getLogger("flag_gems")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(calls)
    scopes = []
    def scope():
        if args.scoped_adapter:
            from silu_policy import Environment, SiluPolicy
            from silu_training_adapter import SiluAdapter
            import importlib.metadata
            import subprocess
            gems_root = str(Path(flag_gems.__file__).resolve().parents[2])
            revision = subprocess.check_output(
                ["git", "-c", "safe.directory=" + gems_root, "-C", gems_root,
                 "rev-parse", "HEAD"], text=True).strip()
            actual = Environment(torch.npu.get_device_name(0), torch.__version__,
                                 importlib.metadata.version("torch-npu"), revision)
            expected = Environment(args.chip_name, "2.10.0+cpu", "2.10.0",
                                   "f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4")
            policy = SiluPolicy(expected, ((2, 32, 3072), (1, 32, 3072)),
                                ("torch.float32", "torch.float16", "torch.bfloat16"),
                                experimental=True)
            adapter = SiluAdapter(policy, actual, runtime, flag_gems)
            scopes.append(adapter)
            result["adapter_environment"] = actual.__dict__
            return adapter
        return flag_gems.use_gems(include=["silu", "silu_backward"])
    # Dispatcher snapshots plus execution log counters establish actual routing.
    def tables():
        return {op: torch._C._dispatch_dump_table("aten::" + op)
                for op in ("silu", "silu_backward")}
    before = tables()
    result["operator_cases"] = []
    for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 5e-3), (torch.bfloat16, 3e-2)):
        torch.manual_seed(1234)
        x = torch.randn(2, 32, 3072, device="npu:0", dtype=dtype, requires_grad=True)
        dy = torch.randn_like(x)
        y = F.silu(x)
        grad = torch.autograd.grad(y, x, dy)[0]
        counts = calls.counts.copy()
        with scope():
            if not args.scoped_adapter:
                result["registered_ops"] = flag_gems.all_registered_ops()
            else:
                result["registration_mode"] = "temporary per forward/backward invocation"
            actual = F.silu(x)
            actual_grad = torch.autograd.grad(actual, x, dy)[0]
            runtime.synchronize(0)
        delta = calls.counts - counts
        assert delta["GEMS SILU FORWARD"] > 0 and delta["GEMS SILU BACKWARD"] > 0, delta
        torch.testing.assert_close(actual, y, atol=tol, rtol=tol)
        torch.testing.assert_close(actual_grad, grad, atol=tol, rtol=tol)
        assert tables() == before, "Dispatcher not restored"
        restored = F.silu(x)
        restored_grad = torch.autograd.grad(restored, x, dy)[0]
        torch.testing.assert_close(restored, y, atol=0, rtol=0)
        torch.testing.assert_close(restored_grad, grad, atol=0, rtol=0)
        result["operator_cases"].append({"dtype": str(dtype), "status": "pass",
            "forward_max_abs": float((actual-y).abs().max()),
            "gradient_max_abs": float((actual_grad-grad).abs().max()),
            "calls": dict(delta), "dispatcher_restored": True, "tolerance": tol})
    del x, y, grad, actual, actual_grad, restored, restored_grad, dy
    if args.scoped_adapter:
        from dataclasses import replace
        result["native_fallback_cases"] = []
        base_input = torch.randn(2, 32, 3072, device="npu:0", requires_grad=True)
        for reason in ("experimental_disabled", "environment_mismatch", "dtype", "shape", "layout"):
            adapter = scope()
            x = base_input
            if reason == "experimental_disabled":
                adapter.policy = replace(adapter.policy, experimental=False)
            elif reason == "environment_mismatch":
                adapter.environment = replace(adapter.environment, gems_revision="unverified")
            elif reason == "dtype":
                adapter.policy = replace(adapter.policy, dtypes=())
            elif reason == "shape":
                x = base_input[:1]
                adapter.policy = replace(adapter.policy, shapes=((2, 32, 3072),))
            else:
                x = base_input.transpose(1, 2)
            dy = torch.randn_like(x)
            expected = F.silu(x)
            expected_grad, = torch.autograd.grad(expected, x, dy)
            counts = calls.counts.copy()
            with adapter:
                actual = F.silu(x)
                actual_grad, = torch.autograd.grad(actual, x, dy)
                runtime.synchronize(0)
            assert not (calls.counts - counts), "Rejected input reached FlagGems"
            assert adapter.counts[("native", "forward", reason)] == 1
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
            assert tables() == before
            result["native_fallback_cases"].append({"reason": reason, "status": "pass"})
        with scope() as adapter, torch.inference_mode():
            x = base_input.detach()
            counts = calls.counts.copy()
            actual = F.silu(x)
        delta = calls.counts - counts
        assert delta["GEMS SILU FORWARD"] == 1 and delta["GEMS SILU BACKWARD"] == 0
        with torch.inference_mode():
            torch.testing.assert_close(actual, F.silu(x), atol=1e-5, rtol=1e-5)
        assert tables() == before
        result["inference_case"] = {"status": "pass", "calls": dict(delta)}
    if args.operator_only:
        result["status"] = "pass"
        result["adapter_events"] = [list(s.events) for s in scopes]
        return
    torch.npu.empty_cache()
    if args.steps == 0:
        result["status"] = "pass"
        return
    from transformers import AutoModelForCausalLM
    def load():
        torch.manual_seed(1234)
        m = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation="eager")
        return m.to("npu:0").train()
    native, target = load(), load()
    pairs = list(zip(native.named_parameters(), target.named_parameters()))
    for (name, p), (other, q) in pairs:
        assert name == other
        torch.testing.assert_close(p, q, atol=0, rtol=0)
        assert bool(torch.isfinite(p).all())
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-5) for m in (native, target)]
    ids = torch.randint(100, 30000, (1, 32), generator=torch.Generator().manual_seed(999)).to("npu:0")
    result["steps"] = []
    result["model_tolerance"] = {"atol": 1e-5, "rtol": 1e-4}
    sample_before = target.get_input_embeddings().weight[int(ids[0,0])].detach().clone()
    for step in range(args.steps):
        if args.same_state:
            with torch.no_grad():
                for (_, p), (_, q) in pairs:
                    q.copy_(p)
        row = {"step": step, "status": "started"}
        result["steps"].append(row)
        counts = calls.counts.copy()
        losses = []
        for model, opt, enabled in zip((native, target), opts, (False, not args.native_control)):
            opt.zero_grad(set_to_none=True)
            torch.manual_seed(1000+step)
            with scope() if enabled else nullcontext():
                loss = model(input_ids=ids, labels=ids, use_cache=False).loss
                assert bool(torch.isfinite(loss))
                loss.backward()
                runtime.synchronize(0)
                losses.append(loss.detach())
        delta = calls.counts - counts
        row["calls"] = dict(delta)
        if args.native_control:
            assert not delta
        else:
            assert delta["GEMS SILU FORWARD"] > 0 and delta["GEMS SILU BACKWARD"] > 0
        assert tables() == before
        row["loss_native"], row["loss_target"] = [float(x) for x in losses]
        torch.testing.assert_close(losses[1], losses[0], atol=1e-5, rtol=1e-4)
        max_grad, count = 0.0, 0
        for (name, p), (_, q) in pairs:
            assert (p.grad is None) == (q.grad is None), name
            if p.grad is not None:
                torch.testing.assert_close(q.grad, p.grad, atol=1e-5, rtol=1e-4, msg=lambda msg: name+": "+msg)
                max_grad = max(max_grad, float((q.grad-p.grad).abs().max()))
                count += 1
        row.update(gradient_tensors_compared=count, gradient_max_abs=max_grad)
        if args.same_state:
            # Advance only the reference trajectory. Next comparison copies this
            # exact weight state, avoiding optimizer/weight drift in the target.
            torch.nn.utils.clip_grad_norm_(native.parameters(), 1.0, error_if_nonfinite=True)
            opts[0].step()
            runtime.synchronize(0)
            row.update(status="pass", same_state=True, target_optimizer_step=False)
            print(json.dumps(row), flush=True)
            continue
        for model, opt in zip((native, target), opts):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
        runtime.synchronize(0)
        max_param = 0.0
        for (name, p), (_, q) in pairs:
            torch.testing.assert_close(q, p, atol=1e-5, rtol=1e-4, msg=lambda msg: name+": "+msg)
            max_param = max(max_param, float((q-p).abs().max()))
        row.update(parameter_tensors_compared=len(pairs), parameter_max_abs=max_param, status="pass")
        print(json.dumps(row), flush=True)
    result["sampled_parameter_update"] = float((target.get_input_embeddings().weight[int(ids[0,0])]-sample_before).abs().max())
    assert result["sampled_parameter_update"] > 0
    result["dispatcher_restored"] = tables() == before
    assert result["dispatcher_restored"]
    if scopes:
        result["adapter_events"] = [list(s.events) for s in scopes]
    result["status"] = "pass"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--native-control", action="store_true", help="Model comparison uses native vs native; operator gates still exercise Gems")
    p.add_argument("--same-state", action="store_true", help="Copy native weights before every forward; advance native optimizer only")
    p.add_argument("--scoped-adapter", action="store_true", help="Test new per-call experimental adapter, not raw global registration")
    p.add_argument("--operator-only", action="store_true", help="Run operator/admission gates without loading the model")
    p.add_argument("--chip-name", default="Ascend910", help="Explicit expected device name; a mismatch stays native")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.steps < 0 or Path(args.out).exists():
        p.error("Require nonnegative steps and unused output path")
    result = {"status": "fail", "scope": "isolated-process experimental SiLU ATen training registration",
              "native_control": args.native_control, "same_state": args.same_state}
    try:
        run(args, result)
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        with open(args.out, "x") as f:
            json.dump(result, f, indent=2, default=str)
    sys.exit(0 if result["status"] == "pass" else 1)
