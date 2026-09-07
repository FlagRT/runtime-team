#!/usr/bin/env python3
"""P7 FINAL FIX for defect #11 (FlagCX native allreduce OOM on small-VRAM cards).

Root cause (proven by real training run, 2026-08-26):
  uniRunnerAllReduce allocated bytes*nranks of DEVICE memory on EVERY call
  (3GB flat grad buffer x 2 ranks = 6GB on a 24GB card). Step 0's allreduce
  runs before optimizer.step() -> ~13GB free -> 6GB cudaMallocAsync succeeds.
  optimizer.step() then creates fp32 AdamW states (~12GB), so step 1's
  6GB cudaMallocAsync fails OOM. DEVCHECK returns flagcxUnhandledDeviceError
  SILENTLY (no log line), which surfaces as:
      "flagcxUnhandledDeviceError: Call to Device function failed."
      "Last error: Undefined: flagcxComm is not fully initialized."  <- red
      herring: getFlagcxErrorDetailStr() has no comm handle, and
      flagcxGetLastError(NULL) always returns that string.
  rank1 (910C, 64GB HBM) never fails -> hangs waiting for rank0.

Fix:
  Process the buffer in slices (default 128MB, env FLAGCX_AR_SLICE_MB):
  per slice: allgather -> D2H -> host reduce -> H2D. Temp device buffer is
  slice*nranks (256MB default) instead of bytes*nranks (6GB). The per-dtype
  reduce switch is kept verbatim; per-slice `count`/`bytes` are shadowed in
  an inner scope so the untouched switch operates on the current slice.
"""
import sys

OLD_A = '''  // 1) gather all ranks into a temporary device buffer (nranks slices)
  void *tmpDev = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&tmpDev, bytes * nranks,
                                          flagcxMemDevice, stream));
  FLAGCXCHECK(
      uniRunnerAllGather(sendbuff, tmpDev, count, datatype, comm, stream));

  // 2) D2H the gathered buffer, then reduce on host
  char *hostBuf = (char *)malloc(bytes * nranks);
  if (hostBuf == nullptr)
    return flagcxSystemError;
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      hostBuf, tmpDev, bytes * nranks, flagcxMemcpyDeviceToHost, stream,
      nullptr));
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));'''

NEW_A = '''  // Kistich(fix-oom): slice-bounded temp buffers. The one-shot path
  // allocated bytes*nranks on EVERY call: on a 24GB card already holding
  // ~21GB of training state (params + grads + fp32 optimizer states), the
  // second allreduce's cudaMallocAsync fails silently inside DEVCHECK and
  // surfaces as flagcxUnhandledDeviceError ("Call to Device function
  // failed"). Process in slices so the temp footprint stays bounded.
  size_t sliceBytes = 128 << 20; // 128MB per slice by default
  const char *sliceEnv = getenv("FLAGCX_AR_SLICE_MB");
  if (sliceEnv) {
    long sliceMb = atol(sliceEnv);
    if (sliceMb > 0)
      sliceBytes = (size_t)sliceMb << 20;
  }
  size_t sliceCount = sliceBytes / esize;
  if (sliceCount == 0)
    sliceCount = 1;

  void *tmpDev = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(&tmpDev, sliceCount * esize * nranks,
                                          flagcxMemDevice, stream));
  char *hostBuf = (char *)malloc(sliceCount * esize * nranks);
  if (hostBuf == nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
    return flagcxSystemError;
  }

  for (size_t off = 0; off < count; off += sliceCount) {
    size_t n = (count - off) < sliceCount ? (count - off) : sliceCount;
    // 1) gather this slice from all ranks into tmpDev (nranks sub-slices)
    FLAGCXCHECK(uniRunnerAllGather(
        (const char *)sendbuff + off * esize, tmpDev, n, datatype, comm,
        stream));
    // 2) D2H the gathered slice, then reduce on host
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        hostBuf, tmpDev, n * esize * nranks, flagcxMemcpyDeviceToHost, stream,
        nullptr));
    FLAGCXCHECK(deviceAdaptor->streamSynchronize(stream));
    {
      // Shadow count/bytes so the per-dtype reduce switch below (kept
      // verbatim) operates on the current slice.
      size_t count = n;
      size_t bytes = n * esize;'''

OLD_B = '''  // 3) H2D the reduced slice
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      recvbuff, hostBuf, bytes, flagcxMemcpyHostToDevice, stream, nullptr));
  free(hostBuf);
  FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
  return flagcxSuccess;'''

NEW_B = '''      // 3) H2D the reduced slice (bytes is the shadowed slice size)
      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
          (char *)recvbuff + off * esize, hostBuf, bytes,
          flagcxMemcpyHostToDevice, stream, nullptr));
    } // end slice scope (count/bytes shadow)
  }   // end slice loop
  free(hostBuf);
  FLAGCXCHECK(deviceAdaptor->deviceFree(tmpDev, flagcxMemDevice, stream));
  return flagcxSuccess;'''


def patch(path):
    s = open(path).read()
    orig = s
    if "Kistich(fix-oom)" in s:
        print("uni_runner.cc: P7 already applied")
        return
    if OLD_A not in s:
        print("uni_runner.cc: ERROR - block A pattern not found")
        sys.exit(1)
    if OLD_B not in s:
        print("uni_runner.cc: ERROR - block B pattern not found")
        sys.exit(1)
    s = s.replace(OLD_A, NEW_A, 1)
    s = s.replace(OLD_B, NEW_B, 1)
    if s != orig:
        open(path, "w").write(s)
        print("uni_runner.cc: sliced allreduce (fix-oom) written")


if __name__ == "__main__":
    patch(sys.argv[1])
    print("P7 sliced allreduce done")
