import sys, json, subprocess
print("=== L1 CONTAINER SELF-CHECK ===")
print("os-release:", [l for l in open("/etc/os-release").read().splitlines() if l.startswith(("NAME=", "VERSION=", "PRETTY_NAME=", "ID=", "VERSION_ID="))])
print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch IMPORT_FAIL:", type(e).__name__, e)
try:
    import torch_mlu
    print("torch_mlu:", torch_mlu.__version__)
    print("device_count(no --device):", torch.mlu.device_count())
except Exception as e:
    print("torch_mlu IMPORT_FAIL:", type(e).__name__, e)
try:
    import triton
    print("triton:", triton.__version__)
except Exception as e:
    print("triton IMPORT_FAIL:", type(e).__name__, e)
try:
    import flag_gems
    print("flag_gems:", getattr(flag_gems, "__version__", "no __version__"), flag_gems.__file__)
except Exception as e:
    print("flag_gems IMPORT_FAIL:", type(e).__name__, e)
