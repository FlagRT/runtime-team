#!/usr/bin/env python3
"""Patch torch_fl ProcessGroupFlagOS._resolve_view: when FlagCX is the inner
backend and the vendor has no zero-copy device alias (ascend view=None), return
the identity view — FlagCX consumes privateuseone tensors natively via data_ptr,
so no flagos->device view conversion is required for the comm path."""
import sys

old = '''        if prof.view is None:
            raise NotImplementedError(
                f"[ProcessGroupFlagOS] GEMS_VENDOR={vendor!r} selected inner "
                f"backend {backend!r}, but no flagos->device view is implemented "
                f"for it (flagos tensors are not a zero-copy alias of "
                f"'{prof.flagcx_dev}'). Implement the corresponding "
                f"_flagos_to_*_view in torch_fl/csrc/module.cc, or use a FlagCX "
                f"adaptor that consumes privateuseone tensors natively."
            )
        import torch_fl._C as _C'''

new = '''        if prof.view is None:
            # FlagCX consumes privateuseone tensors natively via data_ptr()
            # (no zero-copy device alias needed on the comm path). Return the
            # identity view so flagos tensors flow straight into FlagCX.
            if backend == "flagcx":
                return lambda t: t
            raise NotImplementedError(
                f"[ProcessGroupFlagOS] GEMS_VENDOR={vendor!r} selected inner "
                f"backend {backend!r}, but no flagos->device view is implemented "
                f"for it (flagos tensors are not a zero-copy alias of "
                f"'{prof.flagcx_dev}'). Implement the corresponding "
                f"_flagos_to_*_view in torch_fl/csrc/module.cc, or use a FlagCX "
                f"adaptor that consumes privateuseone tensors natively."
            )
        import torch_fl._C as _C'''

paths = [
    "/workspace/PyTorch-Plugin-FL/torch_fl/comm/process_group.py",
    "/root/tf-venv-integration/lib/python3.12/site-packages/torch_fl/comm/process_group.py",
]
for p in paths:
    try:
        s = open(p).read()
    except Exception as e:
        print(f"[{p}] READ FAIL: {e}")
        sys.exit(1)
    if old not in s:
        print(f"[{p}] PATTERN NOT FOUND")
        sys.exit(1)
    s = s.replace(old, new)
    open(p, "w").write(s)
    print(f"[{p}] OK")

print("ALL TORCH_FL VIEW PATCHES APPLIED")
