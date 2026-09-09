#!/usr/bin/env python3
"""P2 FINAL FIX for defect #10 (FlagCX native allreduce deadlock).

Root cause (proven by P2 trace, run 29630):
  cpuAsyncKernel runs inside a cudaLaunchHostFunc callback and calls
  semaphore->wait() — a spin-wait until all ops complete. While a host
  function is executing, the CUDA driver's callback machinery / internal
  locks are held, so the proxy thread's cudaMemcpyAsync (needed to complete
  the ops) blocks forever. Classic cudaLaunchHostFunc self-deadlock.
  Intermittent because it is a pure timing race: if the proxy's memcpy is
  enqueued before the host func starts spinning, the collective completes.

Fix:
  1. launch_kernel.cc: cpuAsyncKernel only calls signalStart() and returns
     immediately (host funcs must never block).
  2. group.cc: the main thread calls semaphore->wait() right after
     launchHostFunc inside flagcxGroupLaunch — completion is enforced
     synchronously by the caller's thread, which owns no driver locks.
     Data visibility is safe: recv ops call subCounter only after their
     H2D copy is eventQuery-confirmed complete, and no subsequent GPU work
     can be enqueued before the collective call returns to Python.

This also fixes the latent use-after-free of the original design (the host
func held a raw semaphore pointer that could outlive the groupLaunch scope).
"""
import sys

def patch_launch(path):
    s = open(path).read()
    orig = s
    old = '''void cpuAsyncKernel(void *args) {
  flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
  semaphore->signalStart();
  fprintf(stderr, "[P2-WAIT-BEGIN] sem=%p\\n", (void *)semaphore);
  fflush(stderr);
  semaphore->wait();
  fprintf(stderr, "[P2-WAIT-END] sem=%p\\n", (void *)semaphore);
  fflush(stderr);
}'''
    new = '''void cpuAsyncKernel(void *args) {
  flagcxHostSemaphore *semaphore = (flagcxHostSemaphore *)args;
  semaphore->signalStart();
  // Kistich(fix-allreduce-deadlock): never block inside a launchHostFunc
  // callback. The original semaphore->wait() here spins while the CUDA
  // driver holds its callback/lock machinery, which blocks the proxy
  // thread's cudaMemcpyAsync -> op never completes -> wait never returns.
  // Completion is enforced by the main thread in flagcxGroupLaunch.
}'''
    if old in s:
        s = s.replace(old, new)
        print("launch_kernel.cc: cpuAsyncKernel wait() removed")
    elif "never block inside a launchHostFunc" in s:
        print("launch_kernel.cc: fix already applied")
    else:
        print("launch_kernel.cc: ERROR - pattern not found"); sys.exit(1)
    if s != orig:
        open(path, "w").write(s); print("launch_kernel.cc written")

def patch_group(path):
    s = open(path).read()
    orig = s
    old = '''    } else {
      FLAGCXCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                (void *)semaphore.get()));
    }'''
    new = '''    } else {
      FLAGCXCHECK(deviceAdaptor->launchHostFunc(launchStream, cpuAsyncKernel,
                                                (void *)semaphore.get()));
      // Kistich(fix-allreduce-deadlock): the host func only signals the proxy
      // to start; the main thread enforces completion here. Waiting inside
      // the host func deadlocks the driver against the proxy thread.
      semaphore->wait();
    }'''
    if old in s:
        s = s.replace(old, new)
        print("group.cc: main-thread wait() added")
    elif "main thread enforces completion here" in s:
        print("group.cc: fix already applied")
    else:
        print("group.cc: ERROR - pattern not found"); sys.exit(1)
    if s != orig:
        open(path, "w").write(s); print("group.cc written")

if __name__ == "__main__":
    patch_launch(sys.argv[1])
    patch_group(sys.argv[2])
    print("P2 deadlock fix done")
