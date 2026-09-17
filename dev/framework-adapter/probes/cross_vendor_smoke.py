"""Small, synchronous cross-vendor kernel/registration checks; no model loading.

Run native, gems, and registration modes in separate processes. This tests
installed vendor packages, not the Ascend-only PreferGems prototype.
"""
import argparse
import importlib
import importlib.metadata as metadata
import json
import sys


def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--mode", choices=("native", "gems", "registration"), required=True)
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"))
    parser.add_argument("--op", choices=("silu", "rms_norm"))
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    if args.plugin:
        importlib.import_module(args.plugin)
    versions = {}
    for package in ("torch", "torch-br", "flag-gems", "triton"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    device = torch.device(args.device)
    runtime = getattr(torch, device.type)
    runtime.set_device(device)
    emit(stage="environment", versions=versions, device=str(device),
         count=runtime.device_count(), name=runtime.get_device_name(device))
    gems = importlib.import_module("flag_gems") if args.mode != "native" else None
    passed = 0
    with torch.inference_mode():
        dtypes = (getattr(torch, args.dtype),) if args.dtype else (torch.float16, torch.bfloat16)
        for dtype in dtypes:
            for shape in ((1, 128), (7, 512)):
                torch.manual_seed(135)
                cpu = torch.randn(shape).to(dtype)
                wc = torch.randn(shape[-1]).to(dtype)
                x, w = cpu.to(device), wc.to(device)
                runtime.synchronize(device)
                for op in ((args.op,) if args.op else ("silu", "rms_norm")):
                    ref = (F.silu(cpu.float()) if op == "silu" else
                           cpu.float() * torch.rsqrt(cpu.float().square().mean(-1, keepdim=True) + 1e-6) * wc.float())
                    def native():
                        return F.silu(x) if op == "silu" else F.rms_norm(x, [shape[-1]], w, 1e-6)
                    hits = set()
                    def trace(frame, event, arg):
                        if (event == "call" and "flag_gems" in frame.f_code.co_filename
                                and frame.f_code.co_name == op):
                            hits.add(frame.f_code.co_filename)
                    if args.mode == "native":
                        actual = native()
                    elif args.mode == "gems":
                        actual = gems.silu(x) if op == "silu" else gems.rms_norm(x, [shape[-1]], w, 1e-6)
                    else:
                        with gems.use_gems(include=[op]):
                            sys.setprofile(trace)
                            try:
                                actual = native()
                                runtime.synchronize(device)
                            finally:
                                sys.setprofile(None)
                        if not hits:
                            raise AssertionError(f"{op}: registration did not enter FlagGems function")
                    runtime.synchronize(device)
                    tol = 0.03 if dtype == torch.bfloat16 else 0.005
                    torch.testing.assert_close(actual.cpu().float(), ref, atol=tol, rtol=tol)
                    torch.testing.assert_close(x.cpu(), cpu, atol=0, rtol=0)
                    torch.testing.assert_close(w.cpu(), wc, atol=0, rtol=0)
                    if actual.device != device or actual.dtype != dtype:
                        raise AssertionError("output device/dtype changed")
                    if args.mode == "registration":
                        hits.clear()
                        sys.setprofile(trace)
                        try:
                            restored = native()
                            runtime.synchronize(device)
                        finally:
                            sys.setprofile(None)
                        if hits:
                            raise AssertionError("FlagGems registration remained after context exit")
                        torch.testing.assert_close(restored.cpu().float(), ref, atol=tol, rtol=tol)
                    passed += 1
                    emit(stage="case", mode=args.mode, op=op, shape=shape,
                         dtype=str(dtype), result="pass")
    # Some vendor FlagGems distributions import vLLM internally. Report that
    # dependency separately; it is not a numerical kernel failure.
    vllm_modules = sorted(n for n in sys.modules
                          if n == "vllm" or n.startswith(("vllm.", "vllm_fl")))
    emit(stage="summary", mode=args.mode, passed=passed,
         vllm_imported=bool(vllm_modules), vllm_modules=vllm_modules)


if __name__ == "__main__":
    main()
