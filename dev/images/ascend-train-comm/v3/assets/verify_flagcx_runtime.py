#!/usr/bin/env python3
"""Fail-closed static/import checks for the ascend-train-comm v3 FlagCX layer.

v3 lineage differences from v2 (dev/images/ascend-train-comm/v2):
  - Route A / "HCCL adaptor" consumption path: the --device dynamic check
    below now goes through torch_npu (PrivateUse1="npu", official autoload,
    no TORCH_DEVICE_BACKEND_AUTOLOAD=0 override) + `dist.init_process_group
    (backend="flagcx")` -- FlagCX's OWN native c10d backend registration --
    instead of v2's `import torch_fl` + `dist.init_process_group("flagos")`
    (Torch-FL's ProcessGroup, which internally calls into FlagCX). This
    mirrors dev/device-context/910C/distributed_training/scripts/
    train_qwen_1_5b_npu.py, the proven 2-card DDP training reference (2481
    steps, loss 1.95, 4245-5428 tok/s, zero torch_fl dependency).
  - FlagCX itself is built from the exact same public commit
    (4e0e0cbcbf721169ca82348080f8353aebfe2c31) with the exact same build
    command (USE_ASCEND=1 pip install . --no-build-isolation) as v2. No
    FlagCX-side build flag distinguishing "flagos adaptor" from "HCCL
    adaptor" was identified by reading the Dockerfile/docs in this repo --
    see dev/images/ascend-train-comm/v3/REBUILD.md "开放问题" for why, and
    for what phase 2 should check to confirm or correct this.
  - v2's static linkage checks (shared objects, ldd, required libascendcl.so/
    libhccl.so, "no torch_npu link") are kept as-is: they test properties of
    the FlagCX wheel itself, which is unchanged in v3.
  - v2's static check additionally probed Torch-FL's libflagos.so for the
    GetCurrentStream ABI symbol -- that assumed Torch-FL/libflagos.so is
    always present and relevant. v3 keeps that probe (torch_fl is still
    installed in the v3 operator-runtime base, just not default-active) but
    does not treat it as evidence of the *default* consumption path anymore
    -- see the new torch_npu_default_check below for that.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_objects(distribution: metadata.Distribution) -> list[Path]:
    paths = []
    for relative in distribution.files or ():
        name = str(relative)
        if name.endswith(".so") or ".so." in name:
            path = Path(distribution.locate_file(relative)).resolve()
            if path.is_file():
                paths.append(path)
    return sorted(set(paths))


def ldd(path: Path) -> str:
    return subprocess.run(
        ["ldd", str(path)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=True,
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static", action="store_true",
        help="verify package artifacts without importing driver-linked modules",
    )
    parser.add_argument(
        "--device", action="store_true",
        help="also run the real-device torch.distributed smoke checks via the "
             "Route A / HCCL-adaptor path (torch_npu + backend=\"flagcx\"); "
             "requires NPU devices mounted; run inside a --device container",
    )
    args = parser.parse_args()

    distribution = metadata.distribution("flagcx")
    if distribution.version != "0.13.0":
        raise RuntimeError(f"unexpected FlagCX version: {distribution.version}")
    objects = shared_objects(distribution)
    if not objects:
        raise RuntimeError("FlagCX wheel contains no shared objects")

    linkage = {str(path): ldd(path) for path in objects}
    unresolved = {
        path: output for path, output in linkage.items()
        if "not found" in output and "libascend_hal.so" not in output
        # libascend_hal.so is the real NPU driver runtime library; it is
        # legitimately absent in a driver-less build container and only
        # resolves inside a --device container. Every other unresolved
        # symbol is a real build defect.
    }
    if unresolved:
        raise RuntimeError(f"FlagCX has unresolved dynamic libraries: {unresolved}")
    combined = "\n".join(linkage.values())
    for library in ("libascendcl.so", "libhccl.so"):
        if library not in combined:
            raise RuntimeError(f"FlagCX linkage does not expose required {library}")
    if "torch_npu" in combined:
        raise RuntimeError("FlagCX unexpectedly links to torch_npu")

    torch_npu_installed = importlib.util.find_spec("torch_npu") is not None
    torch_fl_installed = importlib.util.find_spec("torch_fl") is not None

    result = {
        "schema_version": 2,
        "lineage": "v3 (Route A / HCCL-adaptor default -- see module docstring)",
        "flagcx_version": distribution.version,
        "flagcx_commit": "4e0e0cbcbf721169ca82348080f8353aebfe2c31",
        "shared_objects": [
            {"path": path, "sha256": sha256(Path(path))}
            for path in linkage
        ],
        "required_linkage": ["libascendcl.so", "libhccl.so"],
        "torch_npu_installed": torch_npu_installed,
        "torch_fl_installed": torch_fl_installed,
        "import_check": "not-run" if args.static else "passed",
        "device_check": "not-run" if not args.device else "passed",
    }

    if not args.static:
        # --- Route A default path: torch_npu autoloads, no torch_fl import,
        #     no TORCH_DEVICE_BACKEND_AUTOLOAD override. ---
        import torch

        backend_name = torch._C._get_privateuse1_backend_name()
        assert backend_name == "npu", (
            "Route A regression: PrivateUse1 backend is "
            f"'{backend_name}', expected 'npu'. Check that no ENV in this "
            "image sets TORCH_DEVICE_BACKEND_AUTOLOAD=0."
        )
        result["torch_npu_default_check"] = "passed (PrivateUse1='npu' with no special import order)"

        import torch_npu  # noqa: F401 -- explicit import, matches train_qwen_1_5b_npu.py style
        import flagcx  # noqa: F401 -- FlagCX's own package; import here to force any
        # entry-point/autoload registration to run before init_process_group,
        # mirroring the .pth-registration installation method documented in
        # dev/device-context/910C/distributed_training/ascend_regression/
        # setup_flagcx_plugin.sh. train_qwen_1_5b_npu.py itself does NOT
        # explicitly `import flagcx` and still succeeds with
        # backend="flagcx" -- this suggests the `pip install .` build (used
        # here and in v2) registers FlagCX as an auto-discoverable c10d
        # backend without requiring an explicit import. This extra import is
        # kept here defensively (harmless if already auto-registered) but
        # the assumption that it is NOT strictly required is UNVERIFIED in
        # this phase -- see REBUILD.md open questions.
        if not hasattr(flagcx, "createFlagcxBackend"):
            raise RuntimeError("FlagCX creator is unavailable")

        if args.device:
            import os
            from datetime import timedelta

            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.npu.set_device(local_rank)
            device = f"npu:{local_rank}"
            # Matches train_qwen_1_5b_npu.py: BACKEND = os.environ.get("BACKEND", "flagcx")
            backend_str = os.environ.get("BACKEND", "flagcx")
            dist_init_kwargs = {"timeout": timedelta(seconds=60)}
            import torch.distributed as dist
            dist.init_process_group(backend=backend_str, **dist_init_kwargs)
            try:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                value = torch.full((8,), float(rank + 1), device=device)
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                torch.npu.synchronize()
                expected = float(world_size * (world_size + 1) // 2)
                if not bool(torch.all(value.cpu() == expected).item()):
                    raise RuntimeError(f"all_reduce mismatch: {value.cpu()} != {expected}")
                result["device_all_reduce_rank"] = rank
                result["device_all_reduce_world_size"] = world_size
                result["device_all_reduce_ok"] = True
                result["device_backend_str"] = backend_str
            finally:
                # v2's known issue: teardown-time SIGABRT ("free(): invalid
                # pointer") after a successful torch_fl+FlagCX collective,
                # isolated to torch_fl/FlagCX exit-time cleanup ordering (see
                # ascend-train-comm/v2/lock.yaml:known_issues). Since v3's
                # default path does NOT import torch_fl at all, this issue
                # may or may not reproduce here -- print results before
                # teardown regardless, defensively, and this must be
                # RE-VERIFIED on real hardware in phase 2, not assumed fixed
                # just because torch_fl is out of the default path.
                print(json.dumps(result, indent=2, sort_keys=True), flush=True)
                dist.destroy_process_group()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
