#!/usr/bin/env python3
"""Fail-closed static/import checks for the Ascend FlagCX candidate image."""

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
    args = parser.parse_args()

    if importlib.util.find_spec("torch_npu") is not None:
        raise RuntimeError("torch_npu must not coexist with the Torch-FL runtime")

    distribution = metadata.distribution("flagcx")
    if distribution.version != "0.13.0":
        raise RuntimeError(f"unexpected FlagCX version: {distribution.version}")
    objects = shared_objects(distribution)
    if not objects:
        raise RuntimeError("FlagCX wheel contains no shared objects")

    linkage = {str(path): ldd(path) for path in objects}
    unresolved = {
        path: output for path, output in linkage.items() if "not found" in output
    }
    if unresolved:
        raise RuntimeError(f"FlagCX has unresolved dynamic libraries: {unresolved}")
    combined = "\n".join(linkage.values())
    for library in ("libflagos.so", "libascendcl.so", "libhccl.so"):
        if library not in combined:
            raise RuntimeError(f"FlagCX linkage does not expose required {library}")
    if "torch_npu" in combined:
        raise RuntimeError("FlagCX unexpectedly links to torch_npu")
    object_symbols = {
        str(path): subprocess.run(
            ["nm", "-D", str(path)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
        ).stdout
        for path in objects
    }
    forbidden_stream_symbols = (
        "GetCurrentStreamForDevice", "SetCurrentStreamForDevice",
    )
    leaked_stream_symbols = {
        path: symbol
        for path, symbols in object_symbols.items()
        for symbol in forbidden_stream_symbols
        if symbol in symbols
    }
    if leaked_stream_symbols:
        raise RuntimeError(
            "FlagCX still references Torch-FL stream symbols unavailable on "
            f"Ascend: {leaked_stream_symbols}"
        )

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

    result = {
        "schema_version": 1,
        "flagcx_version": distribution.version,
        "shared_objects": [
            {"path": path, "sha256": sha256(Path(path))}
            for path in linkage
        ],
        "required_linkage": ["libflagos.so", "libascendcl.so", "libhccl.so"],
        "torch_npu_present": False,
        "torch_fl_stream_symbol": "GetCurrentStream",
        "unavailable_stream_symbols_absent": True,
        "import_check": "not-run" if args.static else "passed",
    }

    if not args.static:
        import torch_fl  # noqa: F401 - must register PrivateUse1 first
        import flagcx
        import torch
        import torch.distributed as dist

        if not hasattr(torch, "flagos"):
            raise RuntimeError("torch.flagos was not registered")
        if "flagos" not in dist.Backend.backend_list:
            raise RuntimeError("Torch-FL did not register the flagos ProcessGroup")
        if not hasattr(flagcx, "createFlagcxBackend"):
            raise RuntimeError("FlagCX creator is unavailable")
        capabilities = list(dist.Backend.backend_capability.get("flagcx", []))
        if not {"flagos", "privateuseone"}.intersection(capabilities):
            raise RuntimeError(
                f"FlagCX is not registered for the flagos device: {capabilities}"
            )
        result["flagcx_backend_capability"] = capabilities

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
