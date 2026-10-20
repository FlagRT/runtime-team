"""Small sigmoid MLP training regression; synthetic data, not business accuracy."""
import argparse
import copy
from dataclasses import replace
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import traceback
from contextlib import nullcontext
from qualify_unary_npu import Hits
import _adapter_package
from flagrt_adapter import Environment, OperatorPolicy, load_records
from flagrt_adapter.bridge import ElementwiseAdapter


def run(args, report):
    import torch
    import flag_gems
    import triton
    sys.path.insert(0, args.runtime_root)
    import runtime
    runtime.use("ascend")
    runtime.set_device(0)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
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
    policy = OperatorPolicy(env, ((32, 64),), ("torch.float32",), True,
                            load_records(args.records), operator="sigmoid")
    report["environment"] = env.__dict__
    hits = Hits()
    logger = logging.getLogger("flag_gems")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(hits)
    torch.manual_seed(20261018)
    initial = torch.nn.Sequential(torch.nn.Linear(16,64), torch.nn.Sigmoid(), torch.nn.Linear(64,2))
    # Same initial state and same five batches for all trajectories.
    batches = []
    for _ in range(5):
        x = torch.randn(32,16)
        y = (x[:,0] + .5*x[:,1] > 0).long()
        batches.append((x.to("npu:0"), y.to("npu:0")))
    ops = ("sigmoid", "sigmoid_backward")
    before = {op: torch._C._dispatch_dump_table("aten::"+op) for op in ops}
    references = []
    report.update(model="Linear(16,64)-Sigmoid-Linear(64,2)", parameters=sum(p.numel() for p in initial.parameters()),
                  data="seed=20261018, five synthetic 32x16 batches; labels x0+0.5*x1>0", trajectories=[])
    report['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    modes = ("native", "native_repeat") if args.native_repeat_only else ("native", "target", "disabled", "unqualified")
    for mode in modes:
        model = copy.deepcopy(initial).to("npu:0")
        optim = torch.optim.SGD(model.parameters(), lr=.01)
        selected = policy if mode == "target" else replace(policy,
            experimental=(mode != "disabled"), capabilities=())
        rows = []
        for step,(x,y) in enumerate(batches):
            optim.zero_grad(set_to_none=True)
            report['current'] = dict(mode=mode, step=step+1)
            scope = ElementwiseAdapter({"sigmoid":selected}, env, runtime, flag_gems) if mode not in ("native", "native_repeat") else nullcontext()
            count = hits.counts.copy()
            with scope:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits,y)
                loss.backward()
            runtime.synchronize(0)
            grads = [p.grad.detach().cpu().clone() for p in model.parameters()]
            optim.step()
            weights = [p.detach().cpu().clone() for p in model.parameters()]
            actual = (logits.detach().cpu(), loss.detach().cpu(), grads, weights)
            delta = hits.counts-count
            if mode == "target":
                assert delta["GEMS SIGMOID FORWARD"] == 1 and delta["GEMS SIGMOID BACKWARD"] == 1, delta
            else:
                assert not delta, delta
            if mode in ("disabled", "unqualified"):
                reason = "experimental_disabled" if mode == "disabled" else "capability_unverified"
                assert scope.operator_counts[("sigmoid","native","forward",reason)] == 1
            if mode == "native":
                references.append(actual)
            ref = references[step]
            tol = (1e-5,1e-4) if mode == "target" else (0,0)
            for a,b in zip([actual[0],actual[1]]+grads+weights, [ref[0],ref[1]]+ref[2]+ref[3]):
                torch.testing.assert_close(a,b,atol=tol[0],rtol=tol[1])
                assert torch.isfinite(a).all()
            assert before == {op: torch._C._dispatch_dump_table("aten::"+op) for op in ops}
            rows.append(dict(step=step+1, loss=float(actual[1]), calls=dict(delta),
                gradient_max_abs=max(float((a-b).abs().max()) for a,b in zip(grads,ref[2])),
                parameter_max_abs=max(float((a-b).abs().max()) for a,b in zip(weights,ref[3])),status="pass"))
        model.eval()
        with torch.no_grad():
            prediction = model(batches[0][0]).cpu()
        checkpoint = args.out.parent / ("sigmoid-"+mode+".pt")
        if checkpoint.exists():
            raise RuntimeError("Refusing to overwrite checkpoint")
        torch.save({k:v.cpu() for k,v in model.state_dict().items()}, checkpoint)
        reloaded = copy.deepcopy(initial)
        reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        reloaded.to("npu:0").eval()
        with torch.no_grad():
            restored = reloaded(batches[0][0]).cpu()
        torch.testing.assert_close(prediction,restored,atol=0,rtol=0)
        if mode == "native":
            native_prediction = prediction
        torch.testing.assert_close(prediction,native_prediction,atol=tol[0],rtol=tol[1])
        assert any(not torch.equal(a,b.cpu()) for a,b in zip(initial.parameters(),model.parameters())), "No update"
        report["trajectories"].append(dict(mode=mode,steps=rows,reload="pass",
            prediction_max_abs=float((prediction-native_prediction).abs().max()),status="pass"))
        print(json.dumps(report["trajectories"][-1]),flush=True)
    report.update(status="pass", dispatcher_restored=True, atol=1e-5,rtol=1e-4)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime-root",required=True)
    p.add_argument("--records",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--native-repeat-only",action="store_true")
    p.add_argument("--deterministic",action="store_true")
    args = p.parse_args()
    if args.out.exists():
        p.error("Output exists")
    report = {"status":"fail"}
    try:
        run(args,report)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"],flush=True)
    finally:
        with args.out.open("x") as f:
            json.dump(report,f,indent=2)
    print("overall: "+report["status"].upper(),flush=True)
    sys.exit(0 if report["status"]=="pass" else 1)
