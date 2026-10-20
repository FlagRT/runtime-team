"""Real NPU test of the shared adapter, with separately reviewed qualifications."""
import argparse
from dataclasses import replace
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from qualify_unary_npu import Hits
import _adapter_package
from flagrt_adapter import Environment, OperatorPolicy, load_records
from flagrt_adapter.bridge import ElementwiseAdapter


def run(args, report):
    import torch
    import torch.nn.functional as F
    import flag_gems
    import triton
    sys.path.insert(0, args.runtime_root)
    import runtime
    runtime.use("ascend")
    runtime.set_device(0)
    assert torch.npu.device_count() == 1
    root = str(Path(flag_gems.__file__).resolve().parents[2])
    revision = subprocess.check_output(["git", "-c", "safe.directory=" + root,
        "-C", root, "rev-parse", "HEAD"], text=True).strip()
    builds = []
    for name in ("flagtree", "triton-ascend"):
        try:
            builds.append(name + "==" + metadata.version(name))
        except metadata.PackageNotFoundError:
            pass
    env = Environment(torch.npu.get_device_name(0), torch.__version__, metadata.version("torch-npu"),
        revision, os.environ.get("FLAGRT_VERIFIED_IMAGE_ID", ""), triton.__version__, ";".join(builds))
    report["environment"] = env.__dict__
    records = load_records(args.records)
    policies = {name: OperatorPolicy(env, ((1, 32, 3072), (2, 32, 3072)),
        ("torch.float32", "torch.float16", "torch.bfloat16"), True, records, operator=name)
        for name in ("silu", "sigmoid")}
    hits = Hits()
    logger = logging.getLogger("flag_gems")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(hits)
    ops = ("silu", "silu_backward", "sigmoid", "sigmoid_backward")
    def tables():
        return {name: torch._C._dispatch_dump_table("aten::" + name) for name in ops}
    before = tables()
    report["target_cases"], report["fallback_cases"] = [], []
    for name, native in (("silu", F.silu), ("sigmoid", torch.sigmoid)):
        for shape in policies[name].shapes:
            for dtype, tol in ((torch.float32, 1e-5), (torch.float16, 5e-3), (torch.bfloat16, 3e-2)):
                torch.manual_seed(1234)
                x = torch.randn(shape, dtype=dtype, device="npu:0", requires_grad=True)
                dy = torch.randn_like(x)
                ref = native(x)
                ref_grad, = torch.autograd.grad(ref, x, dy)
                count = hits.counts.copy()
                with ElementwiseAdapter(policies, env, runtime, flag_gems) as scope:
                    y = native(x)
                    grad, = torch.autograd.grad(y, x, dy)
                runtime.synchronize(0)
                delta = hits.counts - count
                assert delta[f"GEMS {name.upper()} FORWARD"] == 1, delta
                assert delta[f"GEMS {name.upper()} BACKWARD"] == 1, delta
                torch.testing.assert_close(y, ref, atol=tol, rtol=tol)
                torch.testing.assert_close(grad, ref_grad, atol=tol, rtol=tol)
                assert tables() == before
                report["target_cases"].append(dict(operator=name, shape=shape, dtype=str(dtype),
                    forward_max_abs=float((y-ref).abs().max()), gradient_max_abs=float((grad-ref_grad).abs().max()),
                    tolerance=tol, calls=dict(delta), events=list(scope.events), status="pass"))
        for reason in ("experimental_disabled", "environment_mismatch", "capability_unverified", "dtype", "shape", "layout"):
            policy, actual_env = policies[name], env
            x = torch.randn((1, 32, 3072), device="npu:0", requires_grad=True)
            if reason == "experimental_disabled":
                policy = replace(policy, experimental=False)
            elif reason == "environment_mismatch":
                actual_env = replace(env, compiler_build="changed")
            elif reason == "capability_unverified":
                policy = replace(policy, capabilities=())
            elif reason == "dtype":
                policy = replace(policy, dtypes=())
            elif reason == "shape":
                policy = replace(policy, shapes=())
            else:
                x = x.transpose(1, 2)
            dy = torch.randn_like(x)
            ref = native(x)
            ref_grad, = torch.autograd.grad(ref, x, dy)
            count = hits.counts.copy()
            with ElementwiseAdapter({name: policy}, actual_env, runtime, flag_gems) as scope:
                y = native(x)
                grad, = torch.autograd.grad(y, x, dy)
            runtime.synchronize(0)
            assert not (hits.counts - count)
            assert scope.operator_counts[(name, "native", "forward", reason)] == 1
            torch.testing.assert_close(y, ref, atol=0, rtol=0)
            torch.testing.assert_close(grad, ref_grad, atol=0, rtol=0)
            assert tables() == before
            report["fallback_cases"].append(dict(operator=name, reason=reason, status="pass"))
    report["status"] = "pass"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--records", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        p.error("Output exists")
    report = {"status": "fail"}
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        with args.out.open("x") as f:
            json.dump(report, f, indent=2)
    print(report["status"], flush=True)
    sys.exit(0 if report["status"] == "pass" else 1)
