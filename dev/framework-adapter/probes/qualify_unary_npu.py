"""Explicit isolated NPU qualification, NOT a normal admission bypass.

Runs raw candidate registration, emits evidence only, never writes approved policy.
The caller must coordinate a free device and verify the container image beforehand.
"""
import argparse
from collections import Counter
import importlib.metadata as metadata
import json
import logging
from pathlib import Path
import subprocess
import sys
import traceback


class Hits(logging.Handler):
    def __init__(self):
        super().__init__()
        self.counts = Counter()
    def emit(self, record):
        message = record.getMessage()
        if message in {f"GEMS {op} {stage}" for op in ("SILU", "SIGMOID")
                       for stage in ("FORWARD", "BACKWARD")}:
            self.counts[message] += 1


def run(args, report):
    import torch
    import torch.nn.functional as F
    import triton
    import flag_gems
    sys.path.insert(0, args.runtime_root)
    import runtime
    runtime.use("ascend")
    runtime.set_device(0)
    source = Path(flag_gems.__file__).resolve().parents[2]
    git = ["git", "-c", "safe.directory=" + str(source), "-C", str(source)]
    revision = subprocess.check_output(git + ["rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(git + ["status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise RuntimeError("Modified FlagGems source; qualify only reviewed source")
    if revision != "f7ae8e6b934a33ec1ccaf2c9aae71edf205f8fb4":
        raise RuntimeError("Unexpected FlagGems revision")
    builds = []
    for name in ("flagtree", "triton-ascend"):
        try:
            builds.append(name + "==" + metadata.version(name))
        except metadata.PackageNotFoundError:
            pass
    report["environment"] = dict(chip=torch.npu.get_device_name(0), torch=torch.__version__,
        torch_npu=metadata.version("torch-npu"), gems_revision=revision, image_id=args.image_id,
        triton=triton.__version__, compiler_build=";".join(builds))
    report["paths"] = dict(gems=flag_gems.__file__, triton=triton.__file__, runtime=runtime.__file__)
    report["device_count"] = torch.npu.device_count()
    if report["device_count"] != 1:
        raise RuntimeError("Qualification requires exactly one mapped device")
    from silu_policy import KNOWN_FAILED_IMAGES
    if ("silu", args.image_id) in KNOWN_FAILED_IMAGES:
        raise RuntimeError("Known failed image: no repeated shmem experiment")
    hits = Hits()
    logger = logging.getLogger("flag_gems")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(hits)
    report["cases"] = []
    for op in args.operators:
        native = F.silu if op == "silu" else torch.sigmoid
        ops = (op, op + "_backward")
        for shape in args.shapes:
            for dtype, tolerance in ((torch.float32, 1e-5), (torch.float16, 5e-3), (torch.bfloat16, 3e-2)):
                torch.manual_seed(1234)
                x = torch.randn(shape, device="npu:0", dtype=dtype, requires_grad=True)
                dy = torch.randn_like(x)
                expected = native(x)
                expected_grad, = torch.autograd.grad(expected, x, dy)
                before = {name: torch._C._dispatch_dump_table("aten::" + name) for name in ops}
                count = hits.counts.copy()
                try:
                    runtime.synchronize(0)
                    with torch.no_grad(), flag_gems.use_gems(include=list(ops)):
                        actual = getattr(torch.ops.aten, op).default(x)
                        grad = getattr(torch.ops.aten, op + "_backward").default(
                            dy, actual if op == "sigmoid" else x)
                        runtime.synchronize(0)
                finally:
                    if before != {name: torch._C._dispatch_dump_table("aten::" + name) for name in ops}:
                        raise RuntimeError("Dispatcher not restored")
                delta = hits.counts - count
                assert delta[f"GEMS {op.upper()} FORWARD"] > 0, delta
                assert delta[f"GEMS {op.upper()} BACKWARD"] > 0, delta
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                torch.testing.assert_close(grad, expected_grad, atol=tolerance, rtol=tolerance)
                restored = native(x)
                restored_grad, = torch.autograd.grad(restored, x, dy)
                torch.testing.assert_close(restored, expected, atol=0, rtol=0)
                torch.testing.assert_close(restored_grad, expected_grad, atol=0, rtol=0)
                item = dict(operator=op, shape=shape, dtype=str(dtype), status="pass",
                            forward_max_abs=float((actual-expected).abs().max()),
                            gradient_max_abs=float((grad-expected_grad).abs().max()),
                            tolerance=tolerance, calls=dict(delta), restored=True)
                report["cases"].append(item)
                print(json.dumps(item), flush=True)
    report["status"] = "pass"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--operator", dest="operators", action="append", choices=("silu", "sigmoid"))
    parser.add_argument("--shape", dest="shapes", action="append", help="positive dimensions, e.g. 32,64")
    args = parser.parse_args()
    args.operators = args.operators or ["silu", "sigmoid"]
    try:
        args.shapes = [tuple(map(int, s.split(','))) for s in args.shapes] if args.shapes else [(1,32,3072), (2,32,3072)]
        if any(not s or any(n <= 0 for n in s) for s in args.shapes):
            raise ValueError()
    except ValueError:
        parser.error("Shapes must be positive integer dimensions")
    if args.out.exists():
        parser.error("Output exists; preserve previous evidence")
    report = {"status": "fail", "scope": "isolated candidate qualification, not automatic admission"}
    try:
        run(args, report)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        with args.out.open("x") as file:
            json.dump(report, file, indent=2)
    sys.exit(0 if report["status"] == "pass" else 1)
