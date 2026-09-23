/*
 * hccl_smoke.c — 910C HCCL 双卡 allgather 冒烟测试
 * 作用：绕开 flagcx/torch_fl，直接验证 HCCL 在 910C 上双卡通信是否可用
 * 用法：容器内 gcc 编译后直接运行（fork 双进程，rank0=dev0, rank1=dev1）
 * 若通过 → flagcx 集成层问题；若失败 → HCCL 环境问题
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
// 直接包含所需子头，绕过 acl_op.h→acl_dump.h 在 CANN 9.0.0 的头文件组合问题
#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "hccl/hccl.h"

#define CHECK_ACL(call)                                                       \
  do {                                                                        \
    aclError _e = (call);                                                     \
    if (_e != ACL_SUCCESS) {                                                  \
      fprintf(stderr, "ACL err %d at %s:%d\n", _e, __FILE__, __LINE__);       \
      return -1;                                                              \
    }                                                                         \
  } while (0)
#define CHECK_HCCL(call)                                                      \
  do {                                                                        \
    HcclResult _e = (call);                                                   \
    if (_e != HCCL_SUCCESS) {                                                 \
      fprintf(stderr, "HCCL err %d at %s:%d\n", _e, __FILE__, __LINE__);      \
      return -1;                                                              \
    }                                                                         \
  } while (0)

int run_rank(int rank, int dev, int rfd, int wfd, int use_bf16, int use_i64) {
  CHECK_ACL(aclInit(NULL));
  CHECK_ACL(aclrtSetDevice(dev));
  HcclRootInfo root;
  memset(&root, 0, sizeof(root));
  if (rank == 0) {
    CHECK_HCCL(HcclGetRootInfo(&root));
    write(wfd, &root, sizeof(root));
  } else {
    size_t got = 0;
    while (got < sizeof(root)) {
      ssize_t n = read(rfd, (char *)&root + got, sizeof(root) - got);
      if (n <= 0) break;
      got += n;
    }
    if (got != sizeof(root)) {
      fprintf(stderr, "rank1: rootinfo recv incomplete (%zu)\n", got);
      return -1;
    }
  }
  HcclComm comm;
  CHECK_HCCL(HcclCommInitRootInfo(2, &root, rank, &comm));
  // HCCL buffers must be DEVICE memory (HBM), not host stack!
  HcclDataType dtype = use_bf16 ? HCCL_DATA_TYPE_BFP16
                                : (use_i64 ? HCCL_DATA_TYPE_INT64
                                           : HCCL_DATA_TYPE_FP32);
  size_t esize = use_bf16 ? 2 : (use_i64 ? 8 : 4);
  char h_send[8] = {0};
  char h_recv[16] = {0};
  if (use_bf16) {
    uint16_t v = (rank == 0) ? 0x3F80 : 0x4000;
    memcpy(h_send, &v, 2);
  } else if (use_i64) {
    int64_t v = (rank == 0) ? 100 : 200;
    memcpy(h_send, &v, 8);
  } else {
    float v = (rank == 0) ? 1.0f : 2.0f;
    memcpy(h_send, &v, 4);
  }
  void *d_send = nullptr, *d_recv = nullptr;
  CHECK_ACL(aclrtMalloc(&d_send, esize, ACL_MEM_MALLOC_NORMAL_ONLY));
  CHECK_ACL(aclrtMalloc(&d_recv, 2 * esize, ACL_MEM_MALLOC_NORMAL_ONLY));
  fprintf(stderr, "[SMKDBG] rank%d d_send=%p d_recv=%p (flagcx uses 0x12c1de9c7xxx)\n",
          rank, d_send, d_recv);
  CHECK_ACL(aclrtMemcpy(d_send, esize, h_send, esize, ACL_MEMCPY_HOST_TO_DEVICE));
  aclrtStream stream;
  CHECK_ACL(aclrtCreateStream(&stream));
  HcclResult ag = HcclAllGather(d_send, d_recv, 1, dtype, comm, stream);
  if (ag != HCCL_SUCCESS) {
    fprintf(stderr, "rank%d HcclAllGather FAILED ret=%d\n", rank, (int)ag);
    return -1;
  }
  CHECK_ACL(aclrtSynchronizeStream(stream));
  CHECK_ACL(aclrtMemcpy(h_recv, 2 * esize, d_recv, 2 * esize,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  if (use_bf16) {
    uint16_t a, b;
    memcpy(&a, h_recv, 2);
    memcpy(&b, h_recv + 2, 2);
    printf("rank%d bf16 allgather OK: [0x%04x 0x%04x]\n", rank, a, b);
  } else if (use_i64) {
    int64_t a, b;
    memcpy(&a, h_recv, 8);
    memcpy(&b, h_recv + 8, 8);
    printf("rank%d i64 allgather OK: [%lld %lld]\n", rank, (long long)a,
           (long long)b);
  } else {
    float fa, fb;
    memcpy(&fa, h_recv, 4);
    memcpy(&fb, h_recv + 4, 4);
    printf("rank%d fp32 allgather OK: [%f %f]\n", rank, fa, fb);
  }
  CHECK_HCCL(HcclCommDestroy(comm));
  CHECK_ACL(aclrtDestroyStream(stream));
  CHECK_ACL(aclrtFree(d_send));
  CHECK_ACL(aclrtFree(d_recv));
  CHECK_ACL(aclrtResetDevice(dev));
  CHECK_ACL(aclFinalize());
  return 0;
}

int main(int argc, char **argv) {
  int use_bf16 = (argc > 1 && strcmp(argv[1], "bf16") == 0);
  int use_i64 = (argc > 1 && strcmp(argv[1], "i64") == 0);
  printf("=== HCCL smoke: %s ===\n",
         use_bf16 ? "BFP16" : (use_i64 ? "INT64" : "FP32"));
  int fd[2];
  if (pipe(fd) != 0) {
    perror("pipe");
    return -1;
  }
  pid_t pid = fork();
  if (pid == 0) {
    close(fd[0]);
    int rc = run_rank(0, 0, -1, fd[1], use_bf16, use_i64);
    close(fd[1]);
    _exit(rc);
  } else if (pid > 0) {
    close(fd[1]);
    int rc = run_rank(1, 1, fd[0], -1, use_bf16, use_i64);
    close(fd[0]);
    int st = 0;
    waitpid(pid, &st, 0);
    if (rc != 0 || !WIFEXITED(st) || WEXITSTATUS(st) != 0) {
      fprintf(stderr, "SMOKE TEST FAILED (parent=%d child=%d)\n", rc,
              WEXITSTATUS(st));
      return 1;
    }
    printf("SMOKE TEST PASSED\n");
    return 0;
  }
  perror("fork");
  return -1;
}
