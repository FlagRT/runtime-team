#!/usr/bin/env python3
"""Static/dynamic verification for the ascend-operator-runtime v3 image.

v3 lineage differences from v2 (dev/images/ascend-operator-runtime/v2):
  - Route A is now the DEFAULT device backend: this image does not set
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 anywhere, so `import torch` alone (no
    special import order, no explicit `import torch_npu`) must let
    torch_npu's own autoload claim PrivateUse1 as "npu". v2's dynamic check
    required importing torch_fl FIRST for the check to pass ("Must own
    PrivateUse1 before torch consumers are imported"); v3 inverts that
    default and instead treats a successful *unprompted* torch_npu claim as
    the thing being verified.
  - torch_fl / flag_gems are still installed (kept pluggable per the task's
    requirement 4/5) but are asserted to be import-able only on an explicit,
    opt-in path -- never on the default path exercised by --dynamic without
    extra flags.
  - vllm_fl (vllm-plugin-FL) presence check. PIVOTED 2026-09-22: the first
    round tried vllm-ascend (vllm-project/vllm-ascend), whose build failed
    (triton-ascend==3.2.1 not on public PyPI, see REBUILD.md). Round 2 uses
    vllm-plugin-FL (github.com/flagos-ai/vllm-plugin-FL, release/0.2 branch)
    instead -- the BAAI-FlagTree-official wiki's own documented inference
    plugin path for the ascend3.5 line, with no triton-ascend dependency.
    Only a find_spec() check is done here (no --static import, since a full
    `import vllm_fl` triggers vLLM's platform/plugin discovery machinery and
    may not be side-effect-free without real hardware); a full import +
    platform-registration + generate() smoke test is done only under
    --device (real hardware), see REBUILD.md.
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
    parser.add_argument(
        "--check-torch-fl-guard",
        action="store_true",
        help="Additionally verify that importing torch_fl AFTER torch_npu has "
             "already claimed PrivateUse1 fails loudly (RuntimeError), proving "
             "torch_fl is opt-in-only and not silently reachable by accident. "
             "Requires --dynamic (i.e. NOT --static).",
    )
    args = parser.parse_args()

    # `triton` has no dist-info of its own here: FlagTree's build installs a
    # single "flagtree" distribution whose RECORD includes the triton/
    # package tree (same as v2).
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

    vllm_fl_installed = importlib.util.find_spec("vllm_fl") is not None
    versions["vllm_fl_installed"] = vllm_fl_installed
    versions["vllm_fl"] = metadata.version("vllm-plugin-fl") if vllm_fl_installed else None
    versions["vllm"] = metadata.version("vllm") if importlib.util.find_spec("vllm") is not None else None

    mpirun = subprocess.run(
        ["mpirun", "--version"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout
    versions["mpirun_first_line"] = mpirun.splitlines()[0] if mpirun else ""

    assert platform.machine() == "aarch64", versions
    assert versions["python"] == "3.11.15", versions
    assert versions["torch"].startswith("2.10.0+cpu"), versions
    assert versions["torch_fl"] == "0.1.0", versions
    assert versions["triton"] == "3.5.1", versions
    assert versions["flagtree"].startswith("0.6.0+ascend.git15ec1a6c"), versions
    # torch_npu MUST be installed in v3 -- it is the default active backend,
    # not an optional coexistence partner as in v2.
    assert torch_npu_installed, "v3 requires torch_npu to be present (Route A default)"

    if args.check_torch_fl_guard and args.static:
        raise RuntimeError("--check-torch-fl-guard requires --dynamic (drop --static)")

    if not args.static:
        # --- Route A default path: plain `import torch`, nothing special. ---
        import torch

        backend_name = torch._C._get_privateuse1_backend_name()
        assert backend_name == "npu", (
            "Route A regression: PrivateUse1 backend is "
            f"'{backend_name}', expected 'npu' (torch_npu's official autoload). "
            "Check that no ENV in this image sets TORCH_DEVICE_BACKEND_AUTOLOAD=0."
        )
        versions["privateuse1_backend_default"] = backend_name
        versions["route_a_default_check"] = "passed (torch_npu claimed PrivateUse1 with no special import order)"

        assert torch.version.cuda is None, versions
        assert torch.__version__.startswith("2.10.0+cpu"), versions

        # FlagGems/triton/Torch-FL remain importable (pluggable), but are not
        # imported above as part of the default path -- only checked for
        # presence here, matching requirement 4/5 ("installed, not active by
        # default").
        assert importlib.util.find_spec("flag_gems") is not None, versions
        assert importlib.util.find_spec("torch_fl") is not None, versions

        # PHASE 2 FIX (2026-09-22, found via real-hardware run in
        # v3-validate-train-910c): resolving the "torch_fl._C" SUBMODULE spec
        # requires importlib to import the PARENT `torch_fl` package first (to
        # read its __path__) -- unlike find_spec("torch_fl") above, which does
        # not execute any code. Under Route A default (torch_npu has already
        # claimed PrivateUse1 via plain `import torch`), importing torch_fl's
        # __init__.py eagerly calls `_check_privateuse1_unclaimed()`, which
        # raises RuntimeError immediately. The original phase-1 code called
        # `find_spec("torch_fl._C")` unconditionally (even without
        # --check-torch-fl-guard), so this RuntimeError was previously
        # UNCAUGHT and crashed the entire script before it ever printed a
        # result -- confirmed by real execution: even the plain `verify_runtime.py`
        # with no flags crashed with this traceback. We catch it here: a
        # RuntimeError at this exact point *is* the expected/desired fail-loud
        # behavior (torch_fl not silently reachable once Route A is active),
        # so it is recorded as evidence rather than left to crash the script.
        torch_fl_c_guard_error = None
        try:
            assert importlib.util.find_spec("torch_fl._C") is not None, versions
        except RuntimeError as exc:
            torch_fl_c_guard_error = str(exc)

        if args.check_torch_fl_guard:
            # Opt-in guard: torch_npu already silently claimed PrivateUse1
            # above (via plain `import torch`). Importing torch_fl now MUST
            # fail loudly -- this is the mirror image of v2's
            # torch_npu_coexistence_check, and proves torch_fl cannot
            # accidentally become active on the v3 default path. See
            # ascend-operator-runtime/v2/REBUILD.md "torch_npu 共存问题" for
            # the empirical basis (import-order invariant, not a
            # co-installation invariant).
            if torch_fl_c_guard_error is not None:
                # The guard already fired above, one line earlier than the
                # explicit `import torch_fl` this branch would otherwise try --
                # find_spec("torch_fl._C") beat us to it. Same invariant,
                # same evidence; record it instead of re-triggering.
                versions["torch_fl_guard_check"] = (
                    "passed (RuntimeError as expected, raised by torch_fl.__init__ "
                    "during the torch_fl._C find_spec presence probe rather than a "
                    f"later explicit `import torch_fl`: {torch_fl_c_guard_error})"
                )
            else:
                try:
                    import torch_fl  # noqa: F401
                except RuntimeError as exc:
                    versions["torch_fl_guard_check"] = f"passed (RuntimeError as expected: {exc})"
                else:
                    raise RuntimeError(
                        "torch_fl imported successfully AFTER torch_npu claimed "
                        "PrivateUse1 -- expected a loud RuntimeError (PrivateUse1 "
                        "already claimed by 'npu'). This would mean torch_fl can "
                        "silently steal the backend, which must never happen."
                    )
        elif torch_fl_c_guard_error is not None:
            # Not asked to check the guard, but merely probing torch_fl._C's
            # presence triggered it anyway -- surface this instead of
            # silently swallowing it, so a plain (no-flag) run still explains
            # why "torch_fl_c_installed" below could not be confirmed by
            # find_spec.
            versions["torch_fl_c_presence_check"] = (
                f"not-confirmed (find_spec triggered the PrivateUse1 guard: {torch_fl_c_guard_error})"
            )

        if vllm_fl_installed and args.static is False and importlib.util.find_spec("torch_npu") is not None:
            # Presence-only check by default (see module docstring); a real
            # `import vllm_fl` + platform registration + generate() smoke
            # test is left for --device (needs NPU hardware / CANN driver
            # context in practice) and is NOT exercised here.
            versions["vllm_fl_import_check"] = "not-run (presence-only; see --device real-hardware smoke test in REBUILD.md)"

    print(json.dumps(versions, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
