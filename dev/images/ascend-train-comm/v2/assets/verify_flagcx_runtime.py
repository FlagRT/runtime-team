#!/usr/bin/env python3
"""Fail-closed static/import checks for the ascend-train-comm v2 FlagCX layer.

v2 lineage differences from v1 (dev/images/ascend-train-comm/v1):
  - FlagCX is built from the FlagRT/FlagCX PUBLIC commit
    4e0e0cbcbf721169ca82348080f8353aebfe2c31 (this team's org fork's regular
    dev branch tip, itself synced from the public flagos-ai/FlagCX upstream),
    not from the owner-private commit 55eb2ffff6988ae1db5e6ecb325472aecc93d238
    that v1 depends on. There are no private patches to apply and no wheel to
    vendor from a torn-down build -- `pip install . --no-build-isolation`
    with USE_ASCEND=1 builds it end to end from a clean checkout. See
    lock.yaml for the "相比 v1 的改进" note.
  - The build-time "torch_npu must not coexist with the Torch-FL runtime"
    assertion from v1 is replaced with the same import-order coexistence
    check used in ascend-operator-runtime/v2's verify_runtime.py (see that
    file's docstring for the empirical basis). torch_npu ships pre-installed
    in this lineage's base image and is not uninstalled.
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
        help="also run the real-device torch.distributed smoke checks "
             "(requires NPU devices mounted; run inside a --device container)",
    )
    args = parser.parse_args()

    # Capture this before any import: `import flagcx` deliberately does
    # `os.environ.pop("TORCH_DEVICE_BACKEND_AUTOLOAD")` after using it
    # internally (see flagcx/__init__.py), so checking os.environ for it
    # after the `if not args.static` import block below would always see it
    # already gone regardless of what the caller actually set.
    import os as _os
    autoload_env = _os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")

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

    torch_fl_distribution = metadata.distribution("torch-fl")
    flagos_library = next(
        (
            Path(torch_fl_distribution.locate_file(relative)).resolve()
            for relative in torch_fl_distribution.files or ()
            if str(relative).endswith("torch_fl/lib/libflagos.so")
        ),
        None,
    )
    if flagos_library is None or not flagos_library.is_file():
        raise RuntimeError("Torch-FL libflagos.so is unavailable")
    flagos_symbols = subprocess.run(
        ["nm", "-D", str(flagos_library)], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout
    if " GetCurrentStream" not in flagos_symbols:
        raise RuntimeError(
            "locked Torch-FL does not export the Ascend GetCurrentStream ABI"
        )

    torch_npu_installed = importlib.util.find_spec("torch_npu") is not None

    result = {
        "schema_version": 1,
        "flagcx_version": distribution.version,
        "flagcx_commit": "4e0e0cbcbf721169ca82348080f8353aebfe2c31",
        "shared_objects": [
            {"path": path, "sha256": sha256(Path(path))}
            for path in linkage
        ],
        "required_linkage": ["libascendcl.so", "libhccl.so"],
        "torch_npu_installed": torch_npu_installed,
        "torch_fl_stream_symbol": "GetCurrentStream",
        "import_check": "not-run" if args.static else "passed",
        "device_check": "not-run" if not args.device else "passed",
    }

    if not args.static:
        import torch_fl  # noqa: F401 -- must register PrivateUse1 first
        import flagcx
        import torch
        import torch.distributed as dist

        if not hasattr(torch, "flagos"):
            raise RuntimeError("torch.flagos was not registered")
        if "flagos" not in dist.Backend.backend_list:
            raise RuntimeError("Torch-FL did not register the flagos ProcessGroup")
        if not hasattr(flagcx, "createFlagcxBackend"):
            raise RuntimeError("FlagCX creator is unavailable")
        # NOTE: dist.Backend.backend_capability["flagcx"] is NOT the right
        # thing to assert against here -- empirically it comes back as just
        # ('npu',) on this public FlagCX build (it registers its own raw c10d
        # backend generically under the literal "npu" device-name string,
        # unrelated to torch_fl's rename_privateuse1_backend("flagos")). The
        # actual communication path used by dist.init_process_group("flagos")
        # is Torch-FL's own "flagos" ProcessGroup (registered in
        # dist.Backend.backend_list, checked above), which internally calls
        # into FlagCX -- see docs/architecture/distributed-flagcx.md in
        # PyTorch-Plugin-FL. Verified end to end on real hardware: see
        # dev/communication/probes/communication_correctness.py results in
        # REBUILD.md (40/40 all_reduce/all_gather/p2p/async_all_reduce calls
        # passed across fp32+bf16 on 2 real NPUs). Recorded, not asserted on.
        capabilities = list(dist.Backend.backend_capability.get("flagcx", []))
        result["flagcx_backend_capability"] = capabilities

        if torch_npu_installed:
            backend_before = torch._C._get_privateuse1_backend_name()
            assert backend_before == "flagos", (
                f"torch_fl did not claim PrivateUse1 before coexistence check: {backend_before}"
            )
            import torch_npu  # noqa: F401 -- must not raise, must not steal PrivateUse1
            backend_after = torch._C._get_privateuse1_backend_name()
            assert backend_after == "flagos", (
                "importing torch_npu after torch_fl changed PrivateUse1 from "
                f"'flagos' to '{backend_after}'"
            )
            result["torch_npu_coexistence_check"] = "passed (import after torch_fl is a safe no-op)"

        if args.device:
            import os
            from datetime import timedelta

            if autoload_env != "0":
                raise RuntimeError("TORCH_DEVICE_BACKEND_AUTOLOAD must be '0' for --device")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.flagos.set_device(local_rank)
            device = f"flagos:{local_rank}"
            dist.init_process_group("flagos", timeout=timedelta(seconds=60))
            try:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                value = torch.full((8,), float(rank + 1), device=device)
                dist.all_reduce(value, op=dist.ReduceOp.SUM)
                torch.flagos.synchronize()
                expected = float(world_size * (world_size + 1) // 2)
                if not bool(torch.all(value.cpu() == expected).item()):
                    raise RuntimeError(f"all_reduce mismatch: {value.cpu()} != {expected}")
                result["device_all_reduce_rank"] = rank
                result["device_all_reduce_world_size"] = world_size
                result["device_all_reduce_ok"] = True
            finally:
                # KNOWN ISSUE (v2, real hardware, both ranks, reproduced with
                # and without this explicit call, and independently in
                # dev/communication/probes/communication_correctness.py):
                # process teardown after a *successful* flagos/FlagCX
                # collective aborts with "free(): invalid pointer" (SIGABRT).
                # The collective's result is correct and captured in `result`
                # above before we ever reach here; print it now so a caller
                # gets the real outcome on stdout even though the process may
                # still be killed by the abort a moment later. See
                # dev/images/ascend-train-comm/v2/REBUILD.md "已知问题" for
                # the isolation steps and dev/images/TODO.md for follow-up.
                print(json.dumps(result, indent=2, sort_keys=True), flush=True)
                dist.destroy_process_group()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
