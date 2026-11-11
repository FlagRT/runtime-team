/* CANN aclnn 实测：验证 ①aclnn 设备侧 add 可用（方案1基础） ②设备能否读 host 映射地址（UVA 核心）
 * Test A: aclnnAdd(devA=1.0, devB=2.0) -> devC，期望 3.0
 * Test B: aclnnAdd(hostMapped=5.0, devB=2.0) -> devC，期望 7.0（证明 NPU 真读了 host 内存）
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"

#define N 4096

static void fill_host(float *h, size_t n, float v) { for (size_t i = 0; i < n; i++) h[i] = v; }

static int do_add(const char *name, void *selfPtr, void *otherPtr, void *outPtr, float expected, aclrtStream stream) {
    int64_t dims[1] = {N};
    aclTensor *self  = aclCreateTensor(dims, 1, ACL_FLOAT, NULL, 0, ACL_FORMAT_ND, NULL, 0, selfPtr);
    aclTensor *other = aclCreateTensor(dims, 1, ACL_FLOAT, NULL, 0, ACL_FORMAT_ND, NULL, 0, otherPtr);
    aclTensor *out   = aclCreateTensor(dims, 1, ACL_FLOAT, NULL, 0, ACL_FORMAT_ND, NULL, 0, outPtr);
    float one = 1.0f;
    aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);

    int ret = 0;
    uint64_t wsSize = 0;
    aclOpExecutor *executor = NULL;
    aclnnStatus st = aclnnAddGetWorkspaceSize(self, other, alpha, out, &wsSize, &executor);
    printf("  [%s] aclnnAddGetWorkspaceSize st=%d wsSize=%lu\n", name, st, (unsigned long)wsSize);
    if (st == 0) {
        void *workspace = NULL;
        aclrtMalloc(&workspace, wsSize ? wsSize : 1, ACL_MEM_MALLOC_HUGE_FIRST);
        st = aclnnAdd(workspace, wsSize, executor, stream);
        printf("  [%s] aclnnAdd st=%d\n", name, st);
        aclrtSynchronizeStream(stream);

        float chk[N];
        aclrtMemcpy(chk, N * sizeof(float), outPtr, N * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        int match = (chk[0] == expected && chk[N/2] == expected && chk[N-1] == expected);
        printf("  [%s] result[0]=%f expected=%f -> %s\n", name, chk[0], expected, match ? "PASS" : "FAIL");
        ret = (st == 0 && match);
        aclrtFree(workspace);
    }

    aclDestroyTensor(self); aclDestroyTensor(other); aclDestroyTensor(out); aclDestroyScalar(alpha);
    return ret;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    aclInit(NULL);
    aclrtSetDevice(0);
    aclrtStream stream = NULL;
    aclrtCreateStream(&stream);

    /* 设备内存 devA=1.0, devB=2.0, devC=0 */
    void *devA = NULL, *devB = NULL, *devC = NULL;
    aclrtMalloc(&devA, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devB, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&devC, N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    float ha[N], hb[N];
    fill_host(ha, N, 1.0f); fill_host(hb, N, 2.0f);
    aclrtMemcpy(devA, N*sizeof(float), ha, N*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(devB, N*sizeof(float), hb, N*sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);

    /* Test A: 纯 device add */
    printf("[TestA] aclnnAdd(devA, devB) -> devC, expect 3.0\n");
    do_add("A", devA, devB, devC, 3.0f, stream);

    /* host 锁页内存 = 5.0，注册 + 取 device 映射地址 */
    void *hostPtr = NULL;
    aclrtMallocHost(&hostPtr, N * sizeof(float));
    fill_host((float*)hostPtr, N, 5.0f);
    aclError e = aclrtHostRegisterV2(hostPtr, N * sizeof(float), ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED);
    void *hostDevPtr = NULL;
    aclrtHostGetDevicePointer(hostPtr, &hostDevPtr, 0);
    printf("[TestB] RegisterV2=%d hostDevPtr=%p (host=%p)\n", e, hostDevPtr, hostPtr);

    /* Test B: device 读 host 映射地址 */
    printf("[TestB] aclnnAdd(hostMapped=5.0, devB=2.0) -> devC, expect 7.0\n");
    do_add("B", hostDevPtr, devB, devC, 7.0f, stream);

    aclrtHostUnregister(hostPtr);
    aclrtFreeHost(hostPtr);
    aclrtFree(devA); aclrtFree(devB); aclrtFree(devC);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    printf("DONE\n");
    return 0;
}
