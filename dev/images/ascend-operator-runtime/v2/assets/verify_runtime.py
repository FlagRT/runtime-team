#!/usr/bin/env python3
"""Static/dynamic verification for the ascend-operator-runtime v2 image.

v2 lineage differences from v1 (dev/images/ascend-operator-runtime/v1):
  - triton comes from BAAI-FlagTree官方 source (package `flagtree`, import
    name still `triton`), compiled in-image per the FlagTree ascend3.5
    manual, not from a PyPI `triton==3.5.0` wheel + a separately-fetched
    Huawei `triton-ascend` wheel. There is therefore no `triton-ascend`
    distribution to query via importlib.metadata; the FlagTree-built
    package itself is queried instead.
  - torch_npu is `torch_npu==2.10.0`, and it SHIPS PRE-INSTALLED in the base
    image (harbor.baai.ac.cn/flagtree/flagtree-ascend3.5-...). v1's
    build-time assertion ("torch_npu must not be installed beside torch_fl")
    is INTENTIONALLY DROPPED here -- see the torch_npu_coexistence check
    below and dev/images/ascend-operator-runtime/v2/REBUILD.md for the
    empirical finding it is based on: torch_fl and torch_npu do not
    conflict as *installed packages*; they conflict only if a NON-torch_fl
    caller lets torch_npu's autoload claim PrivateUse1 before torch_fl
    imports. Importing torch_fl first (with TORCH_DEVICE_BACKEND_AUTOLOAD=0)
    and torch_npu afterwards in the SAME process is empirically safe (no
    exception, PrivateUse1 stays "flagos"); the reverse order fails loudly
    with a clear RuntimeError, never silently. So the real invariant is
    process-level import order, not package (non-)installation, and this
    script checks exactly that instead of banning the package.
"""

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

    # `triton` has no dist-info of its own here: FlagTree's build installs a
    # single "flagtree" distribution whose RECORD includes the triton/
    # package tree, so importlib.metadata.version("triton") raises
    # PackageNotFoundError (verified empirically). Read triton.__version__
    # directly instead -- safe even in --static mode, since importing the
    # pure-Python triton package does not touch any driver-linked module
    # (unlike torch_fl/flag_gems, which call aclInit on import).
    import triton

    versions = {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": metadata.version("torch"),
        "torch_fl": metadata.version("torch-fl"),
        "flag_gems": metadata.version("flag-gems"),
        "triton": getattr(triton, "__version__", "unknown"),
        "flagtree": metadata.version("flagtree"),
    }

    torch_npu_installed = importlib.util.find_spec("torch_npu") is not None
    versions["torch_npu_installed"] = torch_npu_installed
    versions["torch_npu"] = metadata.version("torch_npu") if torch_npu_installed else None

    mpirun = subprocess.run(
        ["mpirun", "--version"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout
    versions["mpirun_first_line"] = mpirun.splitlines()[0] if mpirun else ""

    assert platform.machine() == "aarch64", versions
    assert versions["python"] == "3.11.15", versions
    assert versions["torch"].startswith("2.10.0+cpu"), versions
    assert versions["torch_fl"] == "0.1.0", versions
    # Actual measured triton version for the FlagTree triton_v3.5.x build
    # (commit 15ec1a6cbc8d51f597f46459a500e96f3812c58f) is 3.5.1 -- NOT the
    # 3.5.0 the FlagTree manual's version-line table implies. Recorded as
    # measured fact, not assumed from the manual (see lock.yaml).
    assert versions["triton"] == "3.5.1", versions
    assert versions["flagtree"].startswith("0.6.0+ascend.git15ec1a6c"), versions

    if not args.static:
        import torch_fl  # Must own PrivateUse1 before torch consumers are imported.
        import flag_gems
        import torch
        import triton

        assert torch.version.cuda is None, versions
        assert torch.__version__.startswith("2.10.0+cpu"), versions
        assert getattr(triton, "__version__", "unknown") == "3.5.1", versions
        assert importlib.util.find_spec("flag_gems") is not None, versions
        assert importlib.util.find_spec("torch_fl._C") is not None, versions

        # torch_npu coexistence check (see module docstring): importing
        # torch_npu AFTER torch_fl has already claimed PrivateUse1 must be a
        # silent no-op for torch_fl's registration, never an override and
        # never a crash. Skipped when torch_npu isn't installed at all.
        if torch_npu_installed:
            backend_before = torch._C._get_privateuse1_backend_name()
            assert backend_before == "flagos", (
                "torch_fl did not claim PrivateUse1 as 'flagos' before the "
                f"coexistence check: {backend_before}"
            )
            import torch_npu  # noqa: F401 -- must not raise, must not steal PrivateUse1

            backend_after = torch._C._get_privateuse1_backend_name()
            assert backend_after == "flagos", (
                "importing torch_npu after torch_fl changed the PrivateUse1 "
                f"backend from 'flagos' to '{backend_after}' -- this would be a "
                "genuine coexistence regression, not the expected no-op"
            )
            versions["torch_npu_coexistence_check"] = "passed (import after torch_fl is a safe no-op)"

    print(json.dumps(versions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
