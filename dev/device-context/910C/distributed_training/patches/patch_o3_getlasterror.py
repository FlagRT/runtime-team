#!/usr/bin/env python3
"""O3: flagcxGetLastError 存根完善（缺陷 4）。

- 新增 flagcx/core/include/flagcx_errors.h：线程局部错误记录（header-only,
  C++17 inline thread_local）+ setLastError/getLastErrorMessage
- flagcx.cc：flagcxGetErrorString 实现（错误码描述表）+
  flagcxGetLastError 优先返回 TLS 记录
- DEVCHECK：nvidia_adaptor.h / ascend_adaptor.h 失败时写 TLS（不再静默，
  含厂商错误码与错误描述）
- check.h：FLAGCXCHECK/FLAGCXCHECKGOTO 失败时写 TLS

用法：python3 patch_o3_getlasterror.py <FlagCX根> <nvidia|ascend>
"""
import sys, os

ROOT = sys.argv[1].rstrip("/")
PLAT = sys.argv[2] if len(sys.argv) > 2 else "nvidia"
assert PLAT in ("nvidia", "ascend"), "platform must be nvidia|ascend"

ERR_H = """#pragma once
#include "flagcx.h"
#include <cstdarg>
#include <cstdio>
#include <cstring>

// O3: 线程局部最近一次错误记录（flagcxGetLastError 的数据源）。
// header-only（C++17 inline thread_local，多 TU 共享单实例），无需新 .cc。
// DEVCHECK / FLAGCXCHECK 在失败时调用 flagcx::setLastError 写入。

#define FLAGCX_LAST_ERROR_MSG_LEN 512

namespace flagcx {

struct LastErrorRecord {
  int code = flagcxSuccess;
  int vendorCode = 0;
  char msg[FLAGCX_LAST_ERROR_MSG_LEN];
  LastErrorRecord() { msg[0] = '\\0'; }
};

inline thread_local LastErrorRecord tlsLastError;
inline thread_local bool tlsErrorSet = false;

inline flagcxResult_t setLastError(int code, int vendorCode,
                                   const char *file, int line,
                                   const char *fmt, ...) {
  tlsLastError.code = code;
  tlsLastError.vendorCode = vendorCode;
  snprintf(tlsLastError.msg, sizeof(tlsLastError.msg), "%s:%d ", file, line);
  size_t off = strlen(tlsLastError.msg);
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tlsLastError.msg + off, sizeof(tlsLastError.msg) - off, fmt, ap);
  va_end(ap);
  tlsErrorSet = true;
  return (flagcxResult_t)code;
}

inline const char *getLastErrorMessage() {
  return tlsErrorSet ? tlsLastError.msg : nullptr;
}

inline int getLastErrorCode() {
  return tlsErrorSet ? tlsLastError.code : flagcxSuccess;
}

inline int getLastVendorCode() {
  return tlsErrorSet ? tlsLastError.vendorCode : 0;
}

}  // namespace flagcx
"""

def patch(path, pairs, required=True):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        if required:
            print(f"[FAIL] {path}: not found")
            sys.exit(1)
        print(f"[skip] {path}: not found (optional)")
        return
    s = open(p).read()
    for i, (old, new) in enumerate(pairs):
        n = s.count(old)
        if n == 0:
            print(f"[FAIL] {path} #{i}: pattern not found")
            sys.exit(1)
        if n > 1:
            print(f"[FAIL] {path} #{i}: ambiguous ({n})")
            sys.exit(1)
        s = s.replace(old, new, 1)
    open(p, "w").write(s)
    print(f"[ok] {path}: {len(pairs)} replacements")

# 1) 新增 flagcx_errors.h
err_h_path = os.path.join(ROOT, "flagcx/core/include/flagcx_errors.h")
if os.path.exists(err_h_path):
    print("[skip] flagcx_errors.h 已存在")
else:
    open(err_h_path, "w").write(ERR_H)
    print(f"[ok] 新增 flagcx/core/include/flagcx_errors.h")

# 2) flagcx.cc
patch("flagcx/flagcx.cc", [
    ('#include "flagcx.h"\n#include "adaptor.h"',
     '#include "flagcx.h"\n#include "adaptor.h"\n#include "flagcx_errors.h"'),
    ("""const char *flagcxGetErrorString(flagcxResult_t result) {
  // TODO: implement a method to retrieve error string
  return "Not implemented.";
}""",
     """const char *flagcxGetErrorString(flagcxResult_t result) {
  switch (result) {
  case flagcxSuccess:
    return "Success";
  case flagcxUnhandledDeviceError:
    return "Unhandled device error (see flagcxGetLastError)";
  case flagcxSystemError:
    return "System error (errno)";
  case flagcxInternalError:
    return "Internal error";
  case flagcxInvalidArgument:
    return "Invalid argument";
  case flagcxInvalidUsage:
    return "Invalid usage";
  case flagcxRemoteError:
    return "Remote error";
  case flagcxInProgress:
    return "In progress";
  case flagcxUnhandledCCLError:
    return "Unhandled CCL error";
  case flagcxNotSupported:
    return "Not supported";
  default:
    return "Unknown flagcx error";
  }
}"""),
    ("""const char *flagcxGetLastError(flagcxComm_t comm) {
  // TODO: implement a method to retrieve last error string
  if (comm == NULL) {
    return "Undefined: flagcxComm is not fully initialized.";
  }
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->getLastError(comm->homoComm);
  }
  return "Not implemented.";
}""",
     """const char *flagcxGetLastError(flagcxComm_t comm) {
  // 优先返回线程局部最近一次错误（含文件:行/厂商错误码），由 DEVCHECK /
  // FLAGCXCHECK 在失败时写入（flagcx_errors.h）。
  if (flagcx::tlsErrorSet) {
    return flagcx::tlsLastError.msg;
  }
  if (comm == NULL) {
    return "flagcx: no error recorded (comm is not initialized)";
  }
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->getLastError(comm->homoComm);
  }
  return "flagcx: no error recorded";
}"""),
])

# 3) check.h：FLAGCXCHECK / FLAGCXCHECKGOTO 写 TLS
patch("flagcx/service/include/check.h", [
    ('#include "debug.h"\n#include "type.h"',
     '#include "debug.h"\n#include "type.h"\n#include "flagcx_errors.h"'),
    ("""      /* Print the back trace*/
      if (flagcxDebugNoWarn == 0)
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);
      return RES;""",
     """      /* Print the back trace*/
      if (flagcxDebugNoWarn == 0)
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);
      flagcx::setLastError(RES, 0, __FILE__, __LINE__, "check failed: %s",
                           #call);
      return RES;"""),
    ("""      /* Print the back trace*/
      if (flagcxDebugNoWarn == 0)
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);
      goto label;""",
     """      /* Print the back trace*/
      if (flagcxDebugNoWarn == 0)
        INFO(FLAGCX_ALL, "%s:%d -> %d", __FILE__, __LINE__, RES);
      flagcx::setLastError(RES, 0, __FILE__, __LINE__, "check failed: %s",
                           #call);
      goto label;"""),
])

# 4) DEVCHECK：平台相关
if PLAT == "nvidia":
    patch("flagcx/adaptor/include/nvidia_adaptor.h", [
        ('#include "flagcx.h"\n#include "nccl.h"',
         '#include "flagcx.h"\n#include "nccl.h"\n#include "flagcx_errors.h"'),
        ("""#define DEVCHECK(func)                                                         \\
  {                                                                            \\
    int ret = func;                                                            \\
    if (ret != cudaSuccess)                                                    \\
      return flagcxUnhandledDeviceError;                                       \\
  }""",
         """#define DEVCHECK(func)                                                         \\
  {                                                                            \\
    int ret = func;                                                            \\
    if (ret != cudaSuccess) {                                                  \\
      flagcx::setLastError(flagcxUnhandledDeviceError, (int)ret, __FILE__,     \\
                           __LINE__,                                           \\
                           "device call failed: %s (cudaError %d: %s)", #func, \\
                           (int)ret, cudaGetErrorString((cudaError_t)ret));    \\
      return flagcxUnhandledDeviceError;                                       \\
    }                                                                          \\
  }"""),
    ])
else:
    patch("flagcx/adaptor/include/ascend_adaptor.h", [
        ('#include "flagcx.h"',
         '#include "flagcx.h"\n#include "flagcx_errors.h"'),
        ("""#define DEVCHECK(func)                                                         \\
  {                                                                            \\
    int ret = func;                                                            \\
    if (ret != ACL_SUCCESS)                                                    \\
      return flagcxUnhandledDeviceError;                                       \\
  }""",
         """#define DEVCHECK(func)                                                         \\
  {                                                                            \\
    int ret = func;                                                            \\
    if (ret != ACL_SUCCESS) {                                                  \\
      const char *errmsg = aclGetRecentErrMsg();                               \\
      flagcx::setLastError(flagcxUnhandledDeviceError, (int)ret, __FILE__,     \\
                           __LINE__,                                           \\
                           "device call failed: %s (aclError %d: %s)", #func,  \\
                           (int)ret, errmsg ? errmsg : "");                    \\
      return flagcxUnhandledDeviceError;                                       \\
    }                                                                          \\
  }"""),
    ])

print("=== O3 PATCH DONE ===")
