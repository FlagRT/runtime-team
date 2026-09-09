import sys
ROOT = sys.argv[1].rstrip("/")

# 1) launch_kernel.h：HostSemaphore error 字段 + 方法 + wait 检查
p = ROOT + "/flagcx/core/include/launch_kernel.h"
s = open(p).read()
pairs = [
    # 字段
    ("""  std::vector<flagcxEvent_t> events;
  bool frozen; // true during execution phase

  flagcxHostSemaphore() {
    counter = 0;
    frozen = false;""",
     """  std::vector<flagcxEvent_t> events;
  bool frozen; // true during execution phase
  int error;   // O4: proxy 检测到协议错误后置位，wait() 提前退出

  flagcxHostSemaphore() {
    counter = 0;
    frozen = false;
    error = 0;"""),
    # setError/hasError 方法（pollEnd 前）
    ("""  int pollEnd() override {
    return (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) == 0);
  }
  void wait() override {
    int nDone = 0;
    int nOps = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
    while (nDone < nOps) {
      for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
        if (__atomic_load_n(&signals[it->second].first, __ATOMIC_ACQUIRE) ==
            __atomic_load_n(&signals[it->second].second, __ATOMIC_ACQUIRE)) {
          __atomic_fetch_add(&signals[it->second].first, 1, __ATOMIC_RELEASE);
          nDone++;
        }
      }
      sched_yield();
    }
    __atomic_store_n(&counter, 0, __ATOMIC_RELEASE);
    frozen = false; // unfreeze: allow addCounter for next round
  }""",
     """  int pollEnd() override {
    return (__atomic_load_n(&counter, __ATOMIC_ACQUIRE) == 0);
  }
  // O4: proxy 检测到协议/设备错误时置位，wait() 提前退出（暴露而非卡死）
  void setError() { __atomic_store_n(&error, 1, __ATOMIC_RELEASE); }
  int hasError() { return __atomic_load_n(&error, __ATOMIC_ACQUIRE); }
  void wait() override {
    int nDone = 0;
    int nOps = __atomic_load_n(&counter, __ATOMIC_ACQUIRE);
    while (nDone < nOps) {
      if (__atomic_load_n(&error, __ATOMIC_ACQUIRE)) {
        break; // O4: 错误终结——不再等待未完成 op
      }
      for (auto it = stepInfo.begin(); it != stepInfo.end(); ++it) {
        if (__atomic_load_n(&signals[it->second].first, __ATOMIC_ACQUIRE) ==
            __atomic_load_n(&signals[it->second].second, __ATOMIC_ACQUIRE)) {
          __atomic_fetch_add(&signals[it->second].first, 1, __ATOMIC_RELEASE);
          nDone++;
        }
      }
      sched_yield();
    }
    __atomic_store_n(&counter, 0, __ATOMIC_RELEASE);
    frozen = false; // unfreeze: allow addCounter for next round
  }"""),
]
for i, (old, new) in enumerate(pairs):
    if new in s:
        print(f"[skip] launch_kernel.h #{i}")
        continue
    n = s.count(old)
    if n != 1:
        print(f"[FAIL] launch_kernel.h #{i}: matches={n}"); sys.exit(1)
    s = s.replace(old, new, 1)
    print(f"[ok] launch_kernel.h #{i}")
open(p, "w").write(s)

# 2) group.cc：wait 后 hasError 检查
p2 = ROOT + "/flagcx/core/group.cc"
s2 = open(p2).read()
old2 = """      semaphore->wait();
    }
  }"""
new2 = """      semaphore->wait();
      if (semaphore->hasError()) {
        WARN("flagcxGroupLaunch: proxy reported error (semaphore error flag) "
             "-- aborting");
        ret = flagcxInternalError;
        goto fail;
      }
    }
  }"""
if new2 in s2:
    print("[skip] group.cc hasError")
else:
    n = s2.count(old2)
    if n != 1:
        print(f"[FAIL] group.cc: matches={n}"); sys.exit(1)
    s2 = s2.replace(old2, new2, 1)
    print("[ok] group.cc hasError")
open(p2, "w").write(s2)

# 3) proxy.cc：flagcxProxySend/Recv 错误 -> setError（替换 FLAGCXCHECK）
p3 = ROOT + "/flagcx/core/proxy.cc"
s3 = open(p3).read()
pairs3 = [
    ("""              FLAGCXCHECK(flagcxProxySend(resources, op->recvbuff, op->nbytes,
                                  &op->args));""",
     """              flagcxResult_t sres =
                  flagcxProxySend(resources, op->recvbuff, op->nbytes,
                                  &op->args);
              if (sres != flagcxSuccess) {
                WARN("flagcxProxySend failed: %d (opId=%d) -- flagging error",
                     sres, op->args.opId);
                op->args.semaphore->setError();
              }"""),
    ("""              FLAGCXCHECK(flagcxProxyRecv(resources, op->recvbuff, op->nbytes,
                                  &op->args));""",
     """              flagcxResult_t rres =
                  flagcxProxyRecv(resources, op->recvbuff, op->nbytes,
                                  &op->args);
              if (rres != flagcxSuccess) {
                WARN("flagcxProxyRecv failed: %d (opId=%d) -- flagging error",
                     rres, op->args.opId);
                op->args.semaphore->setError();
              }"""),
]
for i, (old, new) in enumerate(pairs3):
    if new in s3:
        print(f"[skip] proxy.cc #{i}")
        continue
    n = s3.count(old)
    if n != 1:
        print(f"[FAIL] proxy.cc #{i}: matches={n}"); sys.exit(1)
    s3 = s3.replace(old, new, 1)
    print(f"[ok] proxy.cc #{i}")
open(p3, "w").write(s3)
print("=== O4 SEM-ERROR DONE ===")
