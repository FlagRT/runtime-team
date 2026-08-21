#!/usr/bin/env python3
"""Patch FlagCX plugin/torch _build_config.py: remove hard torch_npu dependency
for the ascend adaptor branch (FlagOS/torch_fl ecosystem uses torch_fl, not torch_npu)."""
import sys

p = "/workspace/FlagCX/plugin/torch/_build_config.py"
s = open(p).read()

old = '''    elif adaptor_flag == "-DUSE_ASCEND_ADAPTOR":
        import torch_npu
        pytorch_npu_install_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        pytorch_library_path = os.path.join(pytorch_npu_install_path, "lib")
        # CANN toolkit headers must come BEFORE torch_npu bundled third_party
        # ACL headers (torch_npu 2.11.0 bundles newer ACL headers incompatible
        # with CANN 8.5.1).  We also symlink torch_npu's third_party/acl/inc/acl
        # to CANN's acl/ directory (see install.sh), but adding the CANN include
        # path here is a belt-and-suspenders fix for hccl.h etc.
        cann_home = os.environ.get("ASCEND_HOME_PATH", "")
        if cann_home:
            import platform as _pf
            _arch = "aarch64-linux" if _pf.machine().startswith("aarch") else "x86_64-linux"
            _cann_inc = os.path.join(cann_home, _arch, "include")
            if os.path.isdir(_cann_inc):
                include_dirs += [_cann_inc]
        include_dirs += [os.path.join(pytorch_npu_install_path, "include")]
        library_dirs += [pytorch_library_path]
        libs += ["torch_npu"]'''

new = '''    elif adaptor_flag == "-DUSE_ASCEND_ADAPTOR":
        # FlagOS/torch_fl ecosystem: do NOT depend on torch_npu.
        # Use torch cpp_extension paths + CANN toolkit headers (acl/hccl).
        from torch.utils.cpp_extension import include_paths as _tinc, library_paths as _tlib
        include_dirs += list(_tinc())
        library_dirs += list(_tlib())
        cann_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        if os.path.isdir(cann_home):
            include_dirs += [os.path.join(cann_home, "include"),
                             os.path.join(cann_home, "include", "hccl")]
        libs += []'''

if old not in s:
    print("ERROR: ascend branch not found, patch aborted")
    sys.exit(1)

s = s.replace(old, new)
open(p, "w").write(s)
print("OK: _build_config.py ascend branch patched (no torch_npu)")
