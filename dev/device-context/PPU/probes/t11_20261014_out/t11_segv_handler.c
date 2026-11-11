/* t11_segv_handler.c — PPU 退出期段错误的**原生**现场采集器（执行单 T11）
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * 【为什么不用 ctypes 处理器】
 *   先用 `ctypes.CFUNCTYPE` + `sigaction(SA_SIGINFO)` 写过一版（`t11_sigsegv_probe.py`），
 *   它**非空转自检通过**（故意 `ctypes.string_at(0)` 时能抓到 RIP），但对真崩溃
 *   **一次都没触发** —— 原因是该回调是 **libffi 闭包**，在解释器 finalize 阶段随模块一起被回收，
 *   而目标崩溃恰好发生在这个窗口里 ⇒ 处理器已失效（回调已被释放）。
 *   ⇒ 改用**真正的 C 处理器**：`LD_PRELOAD` 进来的 .so 在 `_dl_fini` 全程都还有效。
 *
 * 【采什么】
 *   ① `si_code` / **`si_addr`**（故障内存地址）、`RIP` / `RSP`（故障指令与栈）；
 *   ② `dladdr(RIP)` ⇒ **模块路径 + 模块基址 + 文件内偏移 + 最近符号名** ←「归属」的硬证据；
 *   ③ `backtrace()` 的逐帧 `模块+偏移`（不依赖 gdb；符号名靠 `addr2line` 事后取）。
 *   之后用容器内已有的 `addr2line -f -C -e <module> 0x<offset>` 取函数名。
 *
 * 【纪律】
 *   · 处理器内只用 `write(2)` 输出，末尾 `_exit(139)`：与"真崩溃"退出码同签名，便于与既有 rc 判据比对。
 *   · 标记行 `T11_CRASH_CAPTURED` 用于区分"被采集"与"未被采集"（不许把两者混读）。
 *   · `snprintf`/`backtrace` 非异步信号安全 —— 诊断用途，可接受；万一在处理器内再次崩溃，
 *     结果与"没装处理器"等价（不会更糟）。
 *
 * 【编译与用法】
 *   gcc -shared -fPIC -O1 -g -o t11_segv_handler.so t11_segv_handler.c -ldl
 *   LD_PRELOAD=<abs>/t11_segv_handler.so python3 <载本> ...
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <ucontext.h>
#include <unistd.h>

#define MAXBUF 8192
#define MAXBT 48

static void on_segv(int sig, siginfo_t *si, void *uc) {
    char buf[MAXBUF];
    int n = 0;
    void *rip = NULL;
    void *rsp = NULL;

    if (uc != NULL) {
        ucontext_t *u = (ucontext_t *)uc;
#if defined(__x86_64__)
        rip = (void *)u->uc_mcontext.gregs[REG_RIP];
        rsp = (void *)u->uc_mcontext.gregs[REG_RSP];
#else
        rip = (void *)u->uc_mcontext.pc;
        rsp = (void *)u->uc_mcontext.sp;
#endif
    }

    n += snprintf(buf + n, MAXBUF - n,
                  "\n===== T11_CRASH_CAPTURED (native) =====\n"
                  "signal=%d si_code=%d si_addr=%p\n"
                  "RIP=%p RSP=%p\n",
                  sig, si ? si->si_code : -1, si ? si->si_addr : NULL, rip, rsp);

    Dl_info di;
    memset(&di, 0, sizeof(di));
    if (rip != NULL && dladdr(rip, &di) != 0 && di.dli_fname != NULL) {
        unsigned long off = (unsigned long)((const char *)rip - (const char *)di.dli_fbase);
        n += snprintf(buf + n, MAXBUF - n,
                      "RIP_MODULE=%s\nRIP_BASE=%p\nRIP_OFFSET=0x%lx\nRIP_SYMBOL=%s\n",
                      di.dli_fname, di.dli_fbase, off,
                      di.dli_sname ? di.dli_sname : "(no symbol)");
    } else {
        n += snprintf(buf + n, MAXBUF - n, "RIP_MODULE=(dladdr failed)\n");
    }

    void *bt[MAXBT];
    int m = backtrace(bt, MAXBT);
    n += snprintf(buf + n, MAXBUF - n, "BACKTRACE_FRAMES=%d\n", m);
    for (int i = 0; i < m && n < MAXBUF - 256; i++) {
        Dl_info d2;
        memset(&d2, 0, sizeof(d2));
        if (dladdr(bt[i], &d2) != 0 && d2.dli_fname != NULL) {
            unsigned long o2 = (unsigned long)((const char *)bt[i] - (const char *)d2.dli_fbase);
            n += snprintf(buf + n, MAXBUF - n, "  FRAME#%d module=%s offset=0x%lx symbol=%s\n",
                          i, d2.dli_fname, o2, d2.dli_sname ? d2.dli_sname : "?");
        } else {
            n += snprintf(buf + n, MAXBUF - n, "  FRAME#%d addr=%p module=?\n", i, bt[i]);
        }
    }

    ssize_t w = write(2, buf, (size_t)n);
    (void)w;
    _exit(139);
}

__attribute__((constructor)) static void t11_install(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = on_segv;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL);
    /* 回读校验：能设置 ≠ 已生效（本项目纪律）。读回不一致就显式喊一声。 */
    struct sigaction rb;
    memset(&rb, 0, sizeof(rb));
    sigaction(SIGSEGV, NULL, &rb);
    if (rb.sa_sigaction != on_segv) {
        ssize_t w = write(2, "[t11] WARN: SIGSEGV handler readback mismatch\n", 45);
        (void)w;
    } else {
        ssize_t w = write(2, "[t11] native SIGSEGV handler installed (readback ok)\n", 53);
        (void)w;
    }
}
