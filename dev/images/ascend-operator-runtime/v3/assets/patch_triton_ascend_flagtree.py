#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Patch FlagTree's built-in ascend triton backend to work with torch_fl
(without a hard dependency on torch_npu), for the ascend3.5 / triton_v3.5.x
lineage (dev/images/*/v2).

Why this is a DIFFERENT script from Torch-FL's scripts/patch_triton_ascend.py:

That script was written against the standalone `triton-ascend` wheel (Huawei's
separately-shipped plugin, version 3.2.1, used by dev/images/*/v1) which had a
flat backends/ascend/driver.py + utils.py with direct, hardcoded torch_npu
calls. FlagTree's own third_party/ascend backend (triton 3.5.1, built from
FlagTree source per the BAAI FlagTree ascend3.5 manual) already refactored
that into a pluggable `backend_strategy_registry` (see
triton/backends/ascend/backend_register.py) with named strategies
("torch_npu", "mindspore") selected by triton/backends/ascend/utils.py's
get_backend_func(). Running the old string-replacement script against this
newer file produces mostly "pattern not found" warnings (verified empirically
on the FlagTree triton_v3.5.x build: 1 real change out of ~13 replacement
rules applies, and it does NOT fix the actual torch_npu calls, which now live
inside backend_register.py's "torch_npu" category instead of inline in
driver.py/utils.py). Silently accepting those warnings would give false
confidence: the resulting triton would still hard-call torch.npu.* internally.

This script instead:
  1. Registers a new "torch_fl" category in backend_strategy_registry,
     mirroring every "torch_npu" strategy function, swapping torch.npu.* for
     torch.flagos.* and the one genuinely torch_npu-specific C++ codegen call
     (at_npu::native::allocate_workspace, used by allocate_sync_block_lock)
     for a raw CANN rtMalloc call -- the same substitution
     Torch-FL's own patch script makes for the old wheel.
  2. Points get_backend_func() at "torch_fl" whenever torch_fl has already
     claimed torch's PrivateUse1 key (checked via hasattr(torch, "flagos"),
     set by torch_fl's own `torch._register_device_module("flagos", ...)"),
     falling back to torch_npu/mindspore detection otherwise -- so this
     patch is inert on a non-torch_fl (e.g. official torch_npu) install of
     the same triton package.
  3. Injects the same _flagos_raw_stream() ctypes helper the old script used
     (resolving GetCurrentStream from the already-loaded libflagos.so) for
     the stream lookup, and fixes the one remaining hardcoded torch_npu call
     in driver.py (get_active_torch_device) to derive the device type
     generically instead of hardcoding "npu".
  4. Fixes triton/backends/nvidia/driver.py's CudaDriver.is_active(), which
     otherwise makes triton raise "2 active drivers" and refuse to start on
     this lineage. Root cause (verified empirically on real 910C hardware):
     Torch-FL's own ecosystem-compat shim intentionally monkeypatches
     `torch.cuda.is_available = lambda: flagos.device_count() > 0` as a
     generic "do I have an accelerator" signal for downstream libraries that
     only know to check torch.cuda (see torch_fl/__init__.py). FlagTree's
     top-level triton backend auto-discovery
     (triton/backends/__init__.py:_discover_backends + driver.py:_create_driver)
     naively treats `CudaDriver.is_active()` (= torch.cuda.is_available() and
     torch.version.hip is None) as proof CUDA is really present, so on a CPU
     torch build with torch_fl's shim active it (wrongly) sees BOTH the real
     NPUDriver and a phantom CudaDriver as "active" and refuses to pick one.
     The fix requires `torch.version.cuda is not None` too -- an actual
     CUDA-compiled torch, not just the ecosystem-compat signal -- which is a
     strictly more correct definition of "is CUDA active" and does not touch
     the ascend driver at all.

Usage:
    python3 patch_triton_ascend_flagtree.py [--triton-path /path/to/triton]

Idempotent: safe to re-run (checks for its own marker before inserting).
"""

import argparse
import os
import sys


def find_triton_path():
    try:
        import triton

        return os.path.dirname(triton.__file__)
    except ImportError:
        print("ERROR: triton not found. Specify --triton-path explicitly.", file=sys.stderr)
        sys.exit(1)


def patch_file(filepath, replacements, label):
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return False
    with open(filepath, "r") as f:
        content = f.read()
    original = content
    applied = 0
    for old, new in replacements:
        if old not in content:
            if new in content:
                continue  # already patched
            print(f"  WARNING [{label}]: pattern not found:")
            print(f"    {repr(old[:120])}")
            continue
        content = content.replace(old, new)
        applied += 1
    if content == original:
        print(f"  OK (already patched or nothing to do): {filepath}")
        return applied > 0 or "n/a"
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  PATCHED ({applied} changes): {filepath}")
    return True


TORCH_FL_STRATEGIES_MARKER = "# --- torch_fl strategies (patched for dev/images/*/v2) ---"

TORCH_FL_STRATEGIES_BLOCK = '''

# --- torch_fl strategies (patched for dev/images/*/v2) ---
# Mirrors every "torch_npu" strategy above, swapping torch.npu.* for
# torch.flagos.* (torch_fl's PrivateUse1 device). The one call that is
# genuinely torch_npu-C++-API-specific (allocate_sync_block_lock, which uses
# at_npu::native::allocate_workspace) is replaced with a raw CANN rtMalloc,
# the same substitution Torch-FL's own patch_triton_ascend.py makes for the
# standalone triton-ascend wheel.

_flagos_stream_fn = None


def _flagos_raw_stream(device):
    """Resolve torch_fl's GetCurrentStream from the already-loaded libflagos.so.

    torch_fl runs every aclnn op on a stream it owns (GetDefaultAclStream ->
    aclrtCreateStream); a kernel launched on any other stream (including 0)
    has no ordering against it. Falls back to 0 with a warning only if the
    symbol cannot be resolved.
    """
    global _flagos_stream_fn
    if _flagos_stream_fn is None:
        import ctypes

        try:
            lib = ctypes.CDLL("libflagos.so")
            fn = lib.GetCurrentStream
        except (OSError, AttributeError) as e:
            import warnings

            warnings.warn(
                "triton-ascend(flagtree) could not resolve GetCurrentStream "
                f"from libflagos.so ({e}); falling back to rt stream 0, which "
                "is NOT ordered against torch_fl's aclnn ops."
            )
            _flagos_stream_fn = lambda _d: 0  # noqa: E731
        else:
            fn.restype = ctypes.c_void_p
            fn.argtypes = [ctypes.c_int]
            _flagos_stream_fn = fn
    return _flagos_stream_fn(int(device)) or 0


@backend_strategy_registry.register("torch_fl", "version_hash")
def version_hash():
    import torch
    return [torch.__version__, "torch_fl_shim"]


@backend_strategy_registry.register("torch_fl", "cxx_abi")
def get_torch_fl_cxx_abi():
    import torch
    return 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0


@backend_strategy_registry.register("torch_fl", "type_convert")
def type_convert():
    import torch
    import numpy as np
    return {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.int64,
        torch.bool: np.bool_,
        torch.complex64: np.complex64,
        torch.complex128: np.complex128,
    }


@backend_strategy_registry.register("torch_fl", "get_device_interface")
def get_device_interface():
    import torch
    return torch.flagos


@backend_strategy_registry.register("torch_fl", "get_empty_tensor")
def get_empty_tensor(size):
    import torch
    return torch.empty(size, dtype=torch.int32, device='flagos')


@backend_strategy_registry.register("torch_fl", "get_tensor_params_shape")
def get_tensor_params_shape(*args):
    import torch
    tensor_params = [arg for arg in args if isinstance(arg, torch.Tensor)]
    return [[s for s in t.shape] for t in tensor_params]


@backend_strategy_registry.register("torch_fl", "get_cc_cmd")
def get_cc_cmd(build_pch):
    import torch
    torch_path = os.path.dirname(os.path.realpath(torch.__file__))
    cc_cmd = [
        f"-I{os.path.join(torch_path, 'include')}",
        f"-D_GLIBCXX_USE_CXX11_ABI={get_torch_fl_cxx_abi()}",
    ]
    # No -ltorch_npu: the torch_fl path only needs generic ATen/PrivateUse1
    # symbols, already satisfied by libtorch/libtorch_cpu at link time.
    return cc_cmd


@backend_strategy_registry.register("torch_fl", "get_current_device")
def get_current_device():
    import torch
    return torch.flagos.current_device()


@backend_strategy_registry.register("torch_fl", "set_current_device")
def set_current_device(device_id):
    import torch
    return torch.flagos.set_device(device_id)


@backend_strategy_registry.register("torch_fl", "get_current_stream")
def get_current_stream(device):
    import torch
    if device is None:
        device = torch.flagos.current_device()
    return _flagos_raw_stream(device)


@backend_strategy_registry.register("torch_fl", "header_file")
def header_file(enable_taskqueue):
    # No NPUWorkspaceAllocator.h / OpCommand.h: torch_fl has no taskqueue
    # equivalent (TRITON_ENABLE_TASKQUEUE defaults to false on this line;
    # async_launch below is a plain synchronous call if it is ever hit).
    return \'\'\'#include <ATen/ATen.h>\'\'\'


@backend_strategy_registry.register("torch_fl", "allocate_memory")
def allocate_memory(size, stream):
    # Generic ATen PrivateUse1 allocation -- identical to the torch_npu
    # strategy's implementation, no torch_npu-specific symbol involved.
    return f"workspace_addr_ptr = const_cast<void *>(at::empty({size}, at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte)).storage().data());"


@backend_strategy_registry.register("torch_fl", "allocate_sync_block_lock")
def allocate_sync_block_lock(size, stream):
    # torch_npu's strategy calls at_npu::native::allocate_workspace, a real
    # torch_npu C++ API with no torch_fl equivalent. Use a raw CANN rtMalloc
    # instead (same substitution as Torch-FL's patch for the old wheel).
    return f"rtMalloc(&syncBlockLock_ptr, {size}, RT_MEMORY_HBM);"


@backend_strategy_registry.register("torch_fl", "pre_launch")
def pre_launch(first_call):
    return ""


@backend_strategy_registry.register("torch_fl", "async_launch")
def async_launch(func):
    # TRITON_ENABLE_TASKQUEUE defaults to false on this triton line, so this
    # is not on the default path. If ever enabled, fall back to a plain
    # synchronous call rather than a real async queue (no torch_fl
    # OpCommand-equivalent exists yet).
    return f"{func};"
'''


def patch_backend_register(triton_path):
    fp = os.path.join(triton_path, "backends", "ascend", "backend_register.py")
    if not os.path.exists(fp):
        print(f"  SKIP (not found): {fp}")
        return False
    with open(fp, "r") as f:
        content = f.read()
    if TORCH_FL_STRATEGIES_MARKER in content:
        print(f"  OK (already patched): {fp}")
        return False
    with open(fp, "a") as f:
        f.write(TORCH_FL_STRATEGIES_BLOCK)
    print(f"  PATCHED (appended torch_fl strategies): {fp}")
    return True


def patch_utils_backend_selection(triton_path):
    fp = os.path.join(triton_path, "backends", "ascend", "utils.py")
    replacements = [
        (
            "        if backend_policy is None:\n"
            "            try:\n"
            "                import torch\n"
            "                import torch_npu\n"
            "                backend_policy = \"torch_npu\"\n"
            "            except ImportError:\n"
            "                backend_policy = \"mindspore\"\n",
            "        if backend_policy is None:\n"
            "            try:\n"
            "                import torch\n"
            "                # torch_fl (PrivateUse1 'flagos') and torch_npu\n"
            "                # (PrivateUse1 'npu') cannot both own the key in\n"
            "                # one process; prefer whichever already claimed\n"
            "                # it. See dev/images/ascend-operator-runtime/v2.\n"
            "                if hasattr(torch, \"flagos\"):\n"
            "                    backend_policy = \"torch_fl\"\n"
            "                else:\n"
            "                    import torch_npu\n"
            "                    backend_policy = \"torch_npu\"\n"
            "            except ImportError:\n"
            "                backend_policy = \"mindspore\"\n",
        ),
    ]
    return patch_file(fp, replacements, "utils.py backend selection")


def patch_driver_active_device(triton_path):
    fp = os.path.join(triton_path, "backends", "ascend", "driver.py")
    replacements = [
        (
            "    def get_active_torch_device(self):\n"
            "        import torch\n"
            "        import torch_npu\n"
            "        return torch.device(\"npu\", self.get_current_device())\n",
            "    def get_active_torch_device(self):\n"
            "        import torch\n"
            "        # Derive the device type generically instead of\n"
            "        # hardcoding 'npu': torch_fl's PrivateUse1 device is\n"
            "        # named 'flagos'. See dev/images/ascend-operator-runtime/v2.\n"
            "        device_type = get_backend_func(\"get_device_interface\").__name__.rsplit(\".\", 1)[-1]\n"
            "        return torch.device(device_type, self.get_current_device())\n",
        ),
    ]
    return patch_file(fp, replacements, "driver.py get_active_torch_device")


def patch_nvidia_driver_is_active(triton_path):
    """Fix the "2 active drivers" false positive -- see module docstring
    point 4. `triton_path` is .../site-packages/triton; the nvidia backend
    lives alongside the ascend one under backends/nvidia/."""
    fp = os.path.join(triton_path, "backends", "nvidia", "driver.py")
    replacements = [
        (
            "        try:\n"
            "            import torch\n"
            "            return torch.cuda.is_available() and (torch.version.hip is None)\n"
            "        except ImportError:\n"
            "            return False\n",
            "        try:\n"
            "            import torch\n"
            "            # torch_fl's ecosystem-compat shim monkeypatches\n"
            "            # torch.cuda.is_available() to True on non-CUDA\n"
            "            # builds (generic \"do I have an accelerator\"\n"
            "            # signal); require an actually CUDA-compiled torch\n"
            "            # too. See dev/images/ascend-operator-runtime/v2.\n"
            "            return (\n"
            "                torch.cuda.is_available()\n"
            "                and torch.version.hip is None\n"
            "                and torch.version.cuda is not None\n"
            "            )\n"
            "        except ImportError:\n"
            "            return False\n",
        ),
    ]
    return patch_file(fp, replacements, "nvidia/driver.py is_active")


def main():
    parser = argparse.ArgumentParser(
        description="Patch FlagTree's ascend triton backend for torch_fl compatibility"
    )
    parser.add_argument("--triton-path", default=None)
    args = parser.parse_args()

    triton_path = args.triton_path or find_triton_path()
    print(f"Triton path: {triton_path}")

    if not os.path.isdir(os.path.join(triton_path, "backends", "ascend")):
        print("ERROR: backends/ascend/ not found. Is this the FlagTree-built triton?", file=sys.stderr)
        sys.exit(1)

    print("\n[1/4] Registering torch_fl strategies in backend_register.py ...")
    patch_backend_register(triton_path)

    print("\n[2/4] Patching utils.py backend selection (get_backend_func) ...")
    patch_utils_backend_selection(triton_path)

    print("\n[3/4] Patching driver.py get_active_torch_device ...")
    patch_driver_active_device(triton_path)

    print("\n[4/4] Patching nvidia/driver.py is_active (fix '2 active drivers') ...")
    patch_nvidia_driver_is_active(triton_path)

    print("\nDone. FlagTree's triton-ascend backend now prefers torch_fl when present.")
    print("NOTE: Clear triton kernel cache if you had previously compiled kernels:")
    print("  rm -rf ~/.triton/cache/")


if __name__ == "__main__":
    main()
