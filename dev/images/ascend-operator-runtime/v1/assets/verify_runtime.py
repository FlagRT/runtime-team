#!/usr/bin/env python3
"""Static verification for the Ascend operator runtime image."""

from __future__ import annotations

import argparse
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static",
        action="store_true",
        help="Do not load driver-linked modules during an offline image build.",
    )
    args = parser.parse_args()

    if importlib.util.find_spec("torch_npu") is not None:
        raise RuntimeError("torch_npu must not be installed beside torch_fl")

    versions = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "torch_fl": metadata.version("torch-fl"),
        "flag_gems": metadata.version("flag-gems"),
        "triton": metadata.version("triton"),
        "triton_ascend": metadata.version("triton-ascend"),
    }
    mpirun = subprocess.run(
        ["/usr/local/mpich/bin/mpirun", "--version"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout
    if "4.1.3" not in mpirun:
        raise RuntimeError(f"unexpected MPICH version: {mpirun.splitlines()[:2]}")
    all_reduce = Path(
        "/usr/local/Ascend/ascend-toolkit/latest/tools/hccl_test/bin/all_reduce_test"
    )
    if not all_reduce.is_file():
        raise RuntimeError(f"HCCL AllReduce executable is missing: {all_reduce}")
    linked = subprocess.run(
        ["ldd", str(all_reduce)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
    ).stdout
    if "not found" in linked:
        raise RuntimeError(f"HCCL AllReduce has unresolved libraries:\n{linked}")
    versions.update(mpich="4.1.3", hccl_all_reduce=str(all_reduce))

    assert platform.machine() == "aarch64", versions
    assert platform.python_version() == "3.11.15", versions
    assert versions["torch"].startswith("2.10.0+cpu"), versions
    assert versions["torch_fl"] == "0.1.0", versions
    assert versions["triton"] == "3.5.0", versions
    assert versions["triton_ascend"] == "3.2.1", versions

    if not args.static:
        import torch_fl  # Must own PrivateUse1 before torch consumers are imported.
        import flag_gems
        import torch
        import triton

        assert torch.version.cuda is None, versions
        assert torch.__version__.startswith("2.10.0+cpu"), versions
        assert getattr(triton, "__version__", "unknown") == "3.5.0", versions
        assert importlib.util.find_spec("flag_gems") is not None, versions
        assert importlib.util.find_spec("torch_fl._C") is not None, versions

    print(json.dumps(versions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
