#!/usr/bin/env python3
"""P6: enforce stream ordering before the ACL callback (real fix for stale
data on 910C send path).

P5 was never taken (torch_npu current stream is a real stream, base != NULL),
so the main SubscribeReport path ran. Wire-level P4 dumps prove the ACL
callback fires WITHOUT waiting for prior stream work: every send D2H read
the tensor one collective late (sent 0, then 2, then 11 instead of 2, 11,
floats) -- i.e. the proxy's D2H raced ahead of the tensor-producing H2D
every single time.

Fix: in the main path, call aclrtSynchronizeStream(stream->base) BEFORE
aclrtLaunchCallback. group.cc has already enqueued
  eventRecord(op->event, op->stream) + streamWaitEvent(launchStream, op->event)
so synchronizing launchStream transitively waits for all tensor-producing
work on op->stream. The callback (signalStart) then fires only after the
input is ready, restoring CUDA-legacy-stream-equivalent happens-before.
"""
import sys

def patch(path):
    s = open(path).read()
    orig = s
    old = '''  aclError err =
      aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base);'''
    new = '''  // Kistich(fix-stale-data): observed ACL callback execution does NOT wait
  // for prior work on the stream -- the proxy's D2H read tensors one
  // collective late (stale data on the wire). Explicitly synchronize the
  // stream first: group.cc enqueued streamWaitEvent(launchStream, op->event)
  // chained to eventRecord(op->event, op->stream), so this sync transitively
  // waits for all tensor-producing work before signalStart fires.
  aclError syncErr = aclrtSynchronizeStream(stream->base);
  if (syncErr != ACL_SUCCESS) {
    fprintf(stderr,
            "[SYNC-DBG] aclrtStreamSynchronize aclErr=%d before callback\\n",
            (int)syncErr);
    fflush(stderr);
  }
  aclError err =
      aclrtLaunchCallback(fn, args, ACL_CALLBACK_NO_BLOCK, stream->base);'''
    if old in s:
        s = s.replace(old, new)
        print("cann_adaptor.cc: stream sync before LaunchCallback added")
    elif "stream sync before LaunchCallback added" in s or "[SYNC-DBG] aclrtStreamSynchronize aclErr" in s:
        print("cann_adaptor.cc: P6 already applied")
    else:
        print("cann_adaptor.cc: ERROR - pattern not found"); sys.exit(1)
    if s != orig:
        open(path, "w").write(s)
        print("cann_adaptor.cc written")

if __name__ == "__main__":
    patch(sys.argv[1])
