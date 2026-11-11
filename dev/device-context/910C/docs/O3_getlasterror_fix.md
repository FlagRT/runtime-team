# O3：`flagcxGetLastError` 存根完善（缺陷 4）——设计与实现草案

> 工作项：O3（`flagcxGetLastError` 存根完善，错误诊断）
> 日期：2026-08-31 ｜ 状态：**✅ 已完成并提交上游**（commit `4bbbae5`，FlagRT/FlagCX `kistich/ascend-dev1.0`）
> 背景教训：P7 排障时 `cudaMallocAsync` OOM 被 `DEVCHECK` **静默吞掉**（零日志），报错尾巴 "flagcxComm is not fully initialized" 是 `flagcxGetLastError(NULL)` 的**固定兜底串**——错误诊断能力缺失导致定位耗时数天。

---

## 1. 现状与问题

### 1.1 已知事实（源码调研结论）
- `flagcxGetLastError(comm)` 位于 `flagcx/flagcx.cc:1593` 附近，是 **TODO 存根**：`comm == NULL` 时返回固定串 "comm is not fully initialized"。
- `getFlagcxErrorDetailStr`（`plugin/torch/flagcx/src/backend_flagcx.cpp`）拿到错误码后**无条件调 `flagcxGetLastError(NULL)`** → 永远拿到固定兜底串（红鲱鱼）。
- `DEVCHECK` 宏（`flagcx/adaptor/include/*.h`）：设备调用失败时**静默返回错误码**，不打印、不记录 → 排障只能靠人工加打点。
- `FLAGCXCHECK` 宏：失败 `goto fail`，fail 路径只打印部分错误（`groupEndInternal` 等有 `[HETERO-DBG]` 打点，剥离后更少）。

### 1.2 问题清单
| # | 问题 | 影响 |
|---|------|------|
| 1 | `flagcxGetLastError` 无真实错误记录，返回固定串 | 上层（plugin/用户）拿不到真实错误 |
| 2 | `DEVCHECK` 静默失败 | 设备调用失败无任何痕迹（P7 OOM 教训） |
| 3 | 无线程局部错误上下文 | 多线程/多 comm 场景无法区分错误来源 |
| 4 | 错误码无描述表 | 错误码数字难读 |

---

## 2. 修复设计（参考 NCCL `ncclGetLastError` / CUDA `cudaGetLastError` 模式）

### 2.1 核心机制：线程局部错误记录

```cpp
// flagcx/core/include/flagcx_errors.h（新增）
#pragma once
#include <string.h>

#define FLAGCX_LAST_ERROR_MSG_LEN 512

// 线程局部最近一次错误记录（含调用点与厂商错误码）
struct flagcxLastErrorRecord {
  int    code;              // flagcxResult_t
  int    vendorCode;        // 厂商错误码（CUDA/ACL），0=无
  char   msg[FLAGCX_LAST_ERROR_MSG_LEN];  // 描述 + 文件:行
};

// 线程局部记录存取
flagcxResult_t flagcxSetLastError(int code, int vendorCode,
                                  const char *file, int line,
                                  const char *fmt, ...);
const char *flagcxGetLastErrorMessage(void);   // flagcxGetLastError 的读取源
int         flagcxGetLastErrorCode(void);
void        flagcxClearLastError(void);
```

实现要点（`flagcx.cc` 或新 `flagcx/core/flagcx_errors.cc`）：
```cpp
static __thread flagcxLastErrorRecord tlsLastError;
static __thread bool tlsErrorSet = false;

flagcxResult_t flagcxSetLastError(int code, int vendorCode,
                                  const char *file, int line,
                                  const char *fmt, ...) {
  tlsLastError.code = code;
  tlsLastError.vendorCode = vendorCode;
  snprintf(tlsLastError.msg, sizeof(tlsLastError.msg), "%s:%d ", file, line);
  size_t off = strlen(tlsLastError.msg);
  va_list ap; va_start(ap, fmt);
  vsnprintf(tlsLastError.msg + off, sizeof(tlsLastError.msg) - off, fmt, ap);
  va_end(ap);
  tlsErrorSet = true;
  return (flagcxResult_t)code;
}
```

### 2.2 `flagcxGetLastError` 改造

```cpp
// flagcx.cc —— 替换 TODO 存根
const char *flagcxGetLastError(flagcxComm_t comm) {
  if (tlsErrorSet) {
    return tlsLastError.msg;
  }
  if (comm == NULL) {
    return "flagcxGetLastError: no error recorded";
  }
  // comm 非空时仍以线程局部记录为准（comm 是句柄，不用于检索）
  return "flagcxGetLastError: no error recorded";
}
```

### 2.3 错误码描述表

```cpp
// flagcx.cc —— flagcxResult -> 描述（覆盖 flagcxResult_t 全枚举）
static const char *flagcxResultStr(flagcxResult_t r) {
  switch (r) {
  case flagcxSuccess:            return "Success";
  case flagcxUnhandledCudaError: return "Unhandled CUDA error";
  case flagcxSystemError:        return "System error (errno)";
  case flagcxInternalError:      return "Internal error";
  case flagcxInvalidArgument:    return "Invalid argument";
  case flagcxInvalidUsage:       return "Invalid usage";
  case flagcxRemoteError:        return "Remote error";
  case flagcxInProgress:         return "In progress";
  case flagcxNumResults:         return "Num results";
  case flagcxUnhandledDeviceError: return "Unhandled device error (see vendorCode / last error msg)";
  case flagcxDeviceMemAlloc:     return "Device memory allocation failed";
  case flagcxDeviceMemFree:      return "Device memory free failed";
  case flagcxDeviceStreamCreate: return "Device stream create failed";
  default:                       return "Unknown flagcx error";
  }
}
```

### 2.4 宏改造：DEVCHECK / FLAGCXCHECK 写入记录

```cpp
// 在宏定义处（flagcx/include/debug.h 或 device 宏头）
#define DEVCHECK(call)                                                       \
  do {                                                                       \
    flagcxResult_t err_ = (call);                                            \
    if (err_ != flagcxSuccess) {                                             \
      /* 记录设备调用失败（含厂商错误码，DEVCHECK 不再静默）*/                  \
      const char *errmsg_ = (call的厂商错误描述，见 2.5);                    \
      flagcxSetLastError(err_, getVendorError(), __FILE__, __LINE__,         \
                         "device call failed: %s", #call);                   \
      return err_;                                                           \
    }                                                                        \
  } while (0)
```

> 注：DEVCHECK 的精确展开需以实际源码为准（不同 adaptor 的 DEVCHECK 定义位置不同）。**保持"失败时写入线程局部记录 + 可选 WARN"的语义，不改变返回行为**（避免影响已验证的控制流）。

### 2.5 厂商错误码获取

```cpp
// CUDA 侧：cudaGetLastError() 在 DEVCHECK 失败后立即调用（未被后续调用覆盖）
// CANN 侧：aclGetRecentErrMsg() / aclGetLastErrMsg()（CANN 9.0 有 aclGetRecentErrMsg）
// 统一封装为 adaptor 新字段或 helper：int flagcxGetVendorLastError();
```

---

## 3. 落地文件清单（预期）

| 文件 | 改动 |
|------|------|
| `flagcx/core/include/flagcx_errors.h`（新） | 错误记录结构 + 接口声明 |
| `flagcx/core/flagcx_errors.cc`（新） | TLS 记录实现 |
| `flagcx/flagcx.cc` | `flagcxGetLastError` 改造 + 错误码表 |
| `flagcx/include/flagcx.h` | 声明（若 GetLastError 原型需调整） |
| 各 adaptor 的 DEVCHECK 宏 | 失败时 `flagcxSetLastError`（CUDA/CANN 侧） |
| `flagcx/include/debug.h` 的 FLAGCXCHECK | fail 路径补 `flagcxSetLastError`（可选项） |
| `makefiles/*.mk` 或 Makefile | 新增 flagcx_errors.cc 编译（如需要） |

---

## 4. 验证方案

| 步骤 | 内容 | 预期 |
|------|------|------|
| V1 | 编译通过（4090-1 CUDA + 910C CANN 两侧） | 无 error |
| V2 | 注入已知错误：小显存下跑大 allreduce（复刻 P7 场景） | `flagcxGetLastError` 返回真实信息（cudaMallocAsync 失败 + 文件:行），不再固定串 |
| V3 | 10 轮 `test_ag_hetero.py` 回归 | 无死锁、sum=3.0 全对（不改变正常路径行为） |
| V4 | plugin 侧 `getFlagcxErrorDetailStr` 拿到的错误串可读 | 上层报错信息有效 |

---

## 5. 提交与看板

- 提交：FlagCX `kistich/ascend-dev1.0`（bundle 中转 push）+ OpenI/GitHub 看板 O3 ✅
- 若 DEVCHECK 改造涉及 CUDA/CANN 两侧行为，先 4090-1 单侧验证再同步 910C

---

## 6. 风险与注意

- **不改变已验证控制流**：DEVCHECK 仅增加"记录"副作用，返回行为不变（避免引入新竞态）。
- 厂商错误码读取时机：`DEVCHECK` 失败后**立即**调用（`cudaGetLastError` 会被后续 CUDA 调用覆盖；CANN 用 `aclGetRecentErrMsg` 更稳）。
- 线程局部记录与 proxy 线程：proxy 线程的错误需通过其返回路径传给主线程（当前机制已有 `op->args.semaphore` 等；若暂不覆盖，可接受"主线程错误优先"）。
