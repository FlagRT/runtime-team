"""Read-only ACL initialization diagnostic, independent of Torch/FlagCX."""
import ctypes
import json
import os

acl = ctypes.CDLL("libascendcl.so")
acl.aclInit.argtypes = [ctypes.c_char_p]
acl.aclInit.restype = ctypes.c_int
code = acl.aclInit(None)
result = {"visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"), "acl_init": code}
if code == 0:
    count = ctypes.c_uint()
    result["get_device_count"] = acl.aclrtGetDeviceCount(ctypes.byref(count))
    result["device_count"] = count.value
    result["acl_finalize"] = acl.aclFinalize()
print(json.dumps(result), flush=True)
raise SystemExit(0 if code == 0 else 1)
