/* CANN UVA 实测：验证 aclrtHostRegisterV2 + aclrtHostGetDevicePointer 真能通
 * 目标：证明设备（NPU）能通过映射后的 device 指针直接访问 host 内存
 * 验证手段：把 host 内存映射得到的 device 指针作为 ACL_MEMCPY_DEVICE_TO_DEVICE
 *          的源，拷到一块普通 device 内存，再 D2H 比对内容一致 → 说明 DMA 引擎
 *          能读这个"host 映射地址"（UVA 真通）。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "acl/acl_rt.h"  /* 只引 runtime 接口，绕开 acl.h 里 acl_dump.h 的 acldumpType 宏保护 bug */

static void test_path(const char *name, void *hostPtr, size_t size, int do_register) {
    aclError e;
    void *devPtr = NULL;

    if (do_register) {
        e = aclrtHostRegisterV2(hostPtr, size, ACL_HOST_REG_PINNED | ACL_HOST_REG_MAPPED);
        printf("  [%s] aclrtHostRegisterV2 ret=%d\n", name, e);
        if (e != 0) { printf("  [%s] SKIP (register failed)\n", name); return; }
    }

    e = aclrtHostGetDevicePointer(hostPtr, &devPtr, 0);
    printf("  [%s] aclrtHostGetDevicePointer ret=%d host=%p dev=%p\n", name, e, hostPtr, devPtr);
    if (e != 0 || devPtr == NULL) {
        printf("  [%s] SKIP (get dev ptr failed)\n", name);
        if (do_register) aclrtHostUnregister(hostPtr);
        return;
    }

    /* 关键验证：device DMA 读这个 host 映射地址 */
    void *devBuf = NULL;
    e = aclrtMalloc(&devBuf, size, ACL_MEM_MALLOC_HUGE_FIRST);
    printf("  [%s] aclrtMalloc ret=%d devBuf=%p\n", name, e, devBuf);
    if (e != 0 || devBuf == NULL) { printf("  [%s] SKIP (malloc failed)\n", name); if (do_register) aclrtHostUnregister(hostPtr); return; }
    e = aclrtMemcpy(devBuf, size, devPtr, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    printf("  [%s] aclrtMemcpy D2D(from host-mapped) ret=%d\n", name, e);

    if (e == 0) {
        void *chk = malloc(size);
        memset(chk, 0, size);
        aclrtMemcpy(chk, size, devBuf, size, ACL_MEMCPY_DEVICE_TO_HOST);
        int match = memcmp(hostPtr, chk, size) == 0;
        printf("  [%s] content match: %s\n", name, match ? "YES (UVA WORKS)" : "NO");
        free(chk);
    }
    aclrtFree(devBuf);
    if (do_register) aclrtHostUnregister(hostPtr);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);  /* 无缓冲，定位段错误点 */
    setvbuf(stderr, NULL, _IONBF, 0);
    fprintf(stderr, "[boot] entering main\n");
    aclError e = aclInit(NULL);
    printf("[aclInit] ret=%d\n", e);
    e = aclrtSetDevice(0);
    printf("[aclrtSetDevice] ret=%d\n", e);

    size_t size = 1024 * 1024; // 1MB

    /* 路径1: 普通 malloc + RegisterV2(PINNED|MAPPED) */
    {
        void *h = malloc(size);
        memset(h, 0xAB, size);
        printf("[Path1] malloc + RegisterV2\n");
        test_path("p1", h, size, 1);
        free(h);
    }

    /* 路径2: aclrtMallocHost（锁页）+ RegisterV2 */
    {
        void *h = NULL;
        e = aclrtMallocHost(&h, size);
        printf("[Path2] aclrtMallocHost ret=%d\n", e);
        if (e == 0) {
            memset(h, 0xCD, size);
            test_path("p2", h, size, 1);
            aclrtFreeHost(h);
        }
    }

    /* 路径3: aclrtMallocHost（锁页，不 Register 直接 GetDevicePointer） */
    {
        void *h = NULL;
        e = aclrtMallocHost(&h, size);
        printf("[Path3] aclrtMallocHost(no-register) ret=%d\n", e);
        if (e == 0) {
            memset(h, 0x12, size);
            test_path("p3", h, size, 0);
            aclrtFreeHost(h);
        }
    }

    aclrtResetDevice(0);
    aclFinalize();
    printf("DONE\n");
    return 0;
}
