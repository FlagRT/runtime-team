// Compile against the patched production header and real CANN libraries.
// Link with --wrap for aclrtCreateEvent, aclrtDestroyEvent, aclrtRecordEvent
// and aclrtStreamWaitEvent to count and check the real calls.
#include "event_flagcx.hpp"
#include <cstdio>
#include <cstdlib>

static unsigned created = 0, destroyed = 0, recorded = 0, waited = 0;
static void check(aclError code) {
  if (code != ACL_SUCCESS) {
    std::fprintf(stderr, "ACL error: %d\n", static_cast<int>(code));
    std::exit(1);
  }
}
extern "C" aclError __real_aclrtCreateEvent(aclrtEvent *);
extern "C" aclError __real_aclrtDestroyEvent(aclrtEvent);
extern "C" aclError __real_aclrtRecordEvent(aclrtEvent, aclrtStream);
extern "C" aclError __real_aclrtStreamWaitEvent(aclrtStream, aclrtEvent);
extern "C" aclError __wrap_aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  auto code = __real_aclrtRecordEvent(event, stream);
  check(code);
  ++recorded;
  return code;
}
extern "C" aclError __wrap_aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  auto code = __real_aclrtStreamWaitEvent(stream, event);
  check(code);
  ++waited;
  return code;
}
extern "C" aclError __wrap_aclrtCreateEvent(aclrtEvent *event) {
  auto code = __real_aclrtCreateEvent(event);
  check(code);
  ++created;
  return code;
}
extern "C" aclError __wrap_aclrtDestroyEvent(aclrtEvent event) {
  auto code = __real_aclrtDestroyEvent(event);
  check(code);
  ++destroyed;
  return code;
}
int main() {
  check(aclInit(nullptr));
  check(aclrtSetDevice(0));
  aclrtStream producer = nullptr, consumer = nullptr;
  check(aclrtCreateStream(&producer));
  check(aclrtCreateStream(&consumer));
  for (unsigned i = 0; i < 1000; ++i) {
    c10d::flagcxCannEvent event;
    event.record(reinterpret_cast<flagcxStream_t>(&producer), 0);
    event.block(reinterpret_cast<flagcxStream_t>(&consumer), 0);
    check(aclrtSynchronizeStream(consumer));
  }
  check(aclrtDestroyStream(consumer));
  check(aclrtDestroyStream(producer));
  check(aclrtResetDevice(0));
  check(aclFinalize());
  std::printf("{\"created\":%u,\"destroyed\":%u,\"recorded\":%u,\"waited\":%u,\"iterations\":1000,"
              "\"scope\":\"production_event_real_acl\"}\n", created, destroyed, recorded, waited);
  return (created == 1000 && destroyed == 1000 && recorded == 1000 && waited == 1000) ? 0 : 1;
}
