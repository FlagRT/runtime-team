"""Run FlagGems XPU tests with an isolated compiler-copy profile.

The installed Triton compiler is never overwritten. This harness is intended
for the locked XPU container and is deliberately separate from the package
installation path so the baseline and the experiment can be run in fresh
processes.
"""
import argparse
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from prepare import prepare


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_profile(backend, profile: str, output: Path):
    original = Path(backend.__file__).read_bytes()
    if profile == "original":
        return original

    data = prepare(original, profile)
    output.write_bytes(data)
    spec = importlib.util.spec_from_file_location("isolated_sdnn_compiler", output)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    live_backend = backend.XPUBackend
    for name in ("make_ttxir", "make_llir"):
        setattr(live_backend, name, staticmethod(getattr(module.XPUBackend, name)))
    return original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("original", "gemm", "epilogue", "diagnostic"),
                        default="diagnostic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-early-combine", action="store_true")
    parser.add_argument("tests", nargs=argparse.REMAINDER,
                        help="pytest arguments, for example tests/test_blas_ops.py::test_accuracy_mm")
    args = parser.parse_args()
    if not args.tests:
        parser.error("at least one pytest target is required")

    os.environ.setdefault("FLAGTREE_BACKEND", "xpu")
    if args.profile == "diagnostic":
        os.environ["XPU_SDNN_PRESERVE_TTXIR_ERRORS"] = "1"
        if args.skip_early_combine:
            os.environ["XPU_SDNN_SKIP_EARLY_COMBINE"] = "1"

    import torch
    import triton
    backend = importlib.import_module("triton.backends.xpu.compiler")
    original = Path(backend.__file__).read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        original = load_profile(backend, args.profile, args.output)
        print(f"PROFILE={args.profile}", flush=True)
        print(f"INSTALLED_COMPILER_SHA256={sha256(original)}", flush=True)
        print(f"GENERATED_COMPILER_SHA256={sha256(args.output.read_bytes()) if args.profile != 'original' else 'same-as-installed'}",
              flush=True)
        print(f"TORCH={torch.__version__} TRITON={triton.__version__}", flush=True)
        import pytest
        return int(pytest.main(args.tests))
    finally:
        if Path(backend.__file__).read_bytes() != original:
            raise RuntimeError("installed compiler changed")


if __name__ == "__main__":
    raise SystemExit(main())
