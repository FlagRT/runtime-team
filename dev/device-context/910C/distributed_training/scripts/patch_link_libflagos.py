#!/usr/bin/env python3
"""_build_config.py: ascend 分支链接 torch_fl 的 libflagos.so（GetCurrentStream 符号）"""
P = "/workspace/FlagCX/plugin/torch/_build_config.py"
s = open(P).read()

old = """        cann_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        if os.path.isdir(cann_home):
            include_dirs += [os.path.join(cann_home, "include"),
                             os.path.join(cann_home, "include", "hccl")]
        libs += []"""
new = """        cann_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        if os.path.isdir(cann_home):
            include_dirs += [os.path.join(cann_home, "include"),
                             os.path.join(cann_home, "include", "hccl")]
        # Link torch_fl's libflagos.so: it exports GetCurrentStream/
        # SetCurrentStream used by the ascend stream guard (current-stream
        # semantics for collectives). Without the link the symbols are not
        # visible at runtime (Python extensions load with RTLD_LOCAL).
        try:
            import torch_fl
            tf_install = os.path.dirname(os.path.abspath(torch_fl.__file__))
            tf_lib = os.path.join(tf_install, "lib")
            if os.path.isdir(tf_lib):
                library_dirs += [tf_lib]
                libs += ["flagos"]
        except ImportError:
            pass
        libs += []"""
assert s.count(old) == 1, "ascend libs block not found"
s = s.replace(old, new, 1)
open(P, "w").write(s)
print("OK: ascend 分支链接 libflagos")
