#!/usr/bin/env python3
"""Patch triton_ascend 3.2.1's strategy registry for Torch-FL.

Torch-FL's upstream patch covers the older driver/utils layout.  Version 3.2.1
moved the remaining torch_npu calls into backend_register.py, so patch that
exact, locked layout and fail the image build if it drifts.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-path", type=Path, required=True)
    args = parser.parse_args()
    target = args.triton_path / "backends" / "ascend" / "backend_register.py"
    text = target.read_text(encoding="utf-8")

    replacements = [
        (
            "    import torch\n    import torch_npu\n"
            "    return [torch.version.git_version, torch_npu.version.git_version]",
            "    import torch\n"
            "    return [torch.version.git_version, 'torch_fl_shim']",
            "version hash",
        ),
        (
            "def get_device_interface():\n    import torch\n    return torch.npu",
            "def get_device_interface():\n    import torch_fl  # noqa: F401\n"
            "    import torch\n    return torch.flagos",
            "device interface",
        ),
        (
            "return torch.empty(size, dtype=torch.int32, device='npu')",
            "return torch.empty(size, dtype=torch.int32, device='flagos')",
            "empty tensor device",
        ),
        (
            "def get_cc_cmd(build_pch):\n    import torch\n    import torch_npu\n"
            "    torch_path = os.path.dirname(os.path.realpath(torch.__file__))\n"
            "    torch_npu_path = os.path.dirname(os.path.realpath(torch_npu.__file__))\n"
            "    cc_cmd = [\n"
            "        f\"-I{os.path.join(torch_path, 'include')}\",\n"
            "        f\"-I{os.path.join(torch_npu_path, 'include')}\",\n"
            "        f\"-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx_abi()}\",\n"
            "    ]\n"
            "    if not build_pch:\n"
            "        cc_cmd += [\n"
            "            f\"-L{os.path.join(torch_npu_path, 'lib')}\",\n"
            "            f\"-ltorch_npu\",\n"
            "        ]\n"
            "    return cc_cmd",
            "def get_cc_cmd(build_pch):\n    import torch\n"
            "    torch_path = os.path.dirname(os.path.realpath(torch.__file__))\n"
            "    return [\n"
            "        f\"-I{os.path.join(torch_path, 'include')}\",\n"
            "        f\"-D_GLIBCXX_USE_CXX11_ABI={get_torch_cxx_abi()}\",\n"
            "    ]",
            "compiler flags",
        ),
        (
            "def get_current_device():\n    import torch\n    import torch_npu\n"
            "    return torch.npu.current_device()",
            "def get_current_device():\n    import torch_fl  # noqa: F401\n"
            "    import torch\n    return torch.flagos.current_device()",
            "current device",
        ),
        (
            "def set_current_device(device_id):\n    import torch\n    import torch_npu\n"
            "    return torch.npu.set_device(device_id)",
            "def set_current_device(device_id):\n    import torch_fl  # noqa: F401\n"
            "    import torch\n    return torch.flagos.set_device(device_id)",
            "set device",
        ),
        (
            "def get_current_stream(device):\n    import torch\n    import torch_npu\n"
            "    if device is None:\n        device = torch.npu.current_device()\n"
            "    if hasattr(torch_npu._C, \"_npu_getCurrentRawStreamNoWait\"):\n"
            "        from torch_npu._C import _npu_getCurrentRawStreamNoWait\n"
            "        return _npu_getCurrentRawStreamNoWait(device)\n"
            "    else:\n        from torch_npu._C import _npu_getCurrentRawStream\n"
            "        return _npu_getCurrentRawStream(device)",
            "def get_current_stream(device):\n    import torch_fl  # noqa: F401\n"
            "    import torch\n    if device is None:\n"
            "        device = torch.flagos.current_device()\n"
            "    stream = torch.flagos.current_stream(device)\n"
            "    native = getattr(stream, '_stream', stream)\n"
            "    return int(getattr(native, 'handle'))",
            "current stream",
        ),
        (
            "return f'''#include <ATen/ATen.h>\n"
            "#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>\n"
            "{'#include <torch_npu/csrc/framework/OpCommand.h>' if {enable_taskqueue} else ''}'''",
            "return '''#include <ATen/ATen.h>'''",
            "C++ headers",
        ),
        (
            "return f\"syncBlockLock_ptr = const_cast<void *>(at_npu::native::allocate_workspace({size}, {stream}).storage().data());\"",
            "return f\"ret = rtMalloc(&syncBlockLock_ptr, {size}, RT_MEMORY_HBM);\"",
            "sync block allocation",
        ),
    ]
    for old, new, label in replacements:
        text = replace_once(text, old, new, label)

    forbidden = ("import torch_npu", "torch_npu.__file__", "-ltorch_npu")
    remaining = [token for token in forbidden if token in text]
    if remaining:
        # Other strategy functions can remain only if they can be selected.  In
        # this locked Torch-FL policy none should survive: fail closed on drift.
        raise RuntimeError(f"unpatched torch_npu references remain: {remaining}")
    target.write_text(text, encoding="utf-8")
    print(f"patched {target}")


if __name__ == "__main__":
    main()
