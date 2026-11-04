/* wrfix.c —— PPU 栈「Stream/Event 退出期 weakref UAF」**运行时等价补丁**（止血用）
 *
 * ⚠️ 这不是厂商修复：厂商的正确修法是把 PR #41596 那两行编进 libtorch
 *    （THNPStream_dealloc / THNPEvent_dealloc 在 tp_free 前调 PyObject_ClearWeakRefs）。
 *    本补丁在**进程内**给类型对象换一个 tp_dealloc 包装，语义完全相同：
 *        先 PyObject_ClearWeakRefs(self)，再调原 dealloc。
 *    目的：① 给厂商一个可直接照抄的最小可运行证据；② 验收期间止血（不改镜像、可撤）。
 *
 * 用法（Python 侧）：
 *    import wrfix            # import 即打补丁；返回值/日志见 wrfix.info()
 *    wrfix.info()            # 返回 (stream_patched, event_patched, 原/新地址)
 *
 * 判据（见同目录 run_ppu_wrfix.sh）：
 *    P1 Stream/Event 的 weakref 回调**开始触发**（修前均为 False）
 *    P2 「Stream + weakref + 被回收」进程 rc 139 → 0
 *    P3 功能形态不变（流可用、事件计时可用）
 *    P4 我方验收读数零变化（smoke / 离线自检）
 *
 * 编译：gcc -O2 -shared -fPIC $(python3-config --includes) -o wrfix$(python3-config --extension-suffix) wrfix.c
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* 两个类型各自的原 dealloc（不能共用一个，包装函数需分别回跳） */
static destructor orig_stream = NULL;
static destructor orig_event = NULL;
static int stream_ok = 0;
static int event_ok = 0;

/* 等价于上游修法：对象内存被释放前，先把挂在它身上的 weakref 书签清干净。
 * 若原实现已自行清理（未来厂商修好后），PyObject_ClearWeakRefs 对已清对象是幂等的，
 * 因此本包装可安全叠加，不需要版本判断。 */
static void wr_stream_dealloc(PyObject *self)
{
    PyObject_ClearWeakRefs(self);
    orig_stream(self);
}

static void wr_event_dealloc(PyObject *self)
{
    PyObject_ClearWeakRefs(self);
    orig_event(self);
}

/* 静态类型对象可能落在只读页（PyType_Freeze / 只读 .data）；为稳妥先临时放开写权限。
 * ⚠️ 不还原写保护：页面可能同时承载其他只读常量，还原反而有风险；本补丁仅在进程内生效，
 *    进程退出即消失，故接受"进程内该页可写"这一副作用（已在文档里登记）。 */
static int make_writable(void *addr, size_t len)
{
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return -1;
    uintptr_t start = (uintptr_t)addr & ~((uintptr_t)ps - 1);
    uintptr_t end = ((uintptr_t)addr + len + (uintptr_t)ps - 1) & ~((uintptr_t)ps - 1);
    return mprotect((void *)start, (size_t)(end - start), PROT_READ | PROT_WRITE);
}

static int patch_one(const char *modname, const char *attr, const char *tag,
                     destructor wrapper, destructor *orig_slot, int *ok_slot)
{
    PyObject *mod = PyImport_ImportModule(modname);
    if (!mod) return -1;
    PyObject *cls = PyObject_GetAttrString(mod, attr);
    Py_DECREF(mod);
    if (!cls) {
        PyErr_Clear();
        fprintf(stderr, "[wrfix] %s: 找不到 %s.%s（跳过）\n", tag, modname, attr);
        return 0;
    }
    if (!PyType_Check(cls)) {
        fprintf(stderr, "[wrfix] %s: 不是类型对象（跳过）\n", tag);
        Py_DECREF(cls);
        return 0;
    }
    PyTypeObject *tp = (PyTypeObject *)cls;

    if (tp->tp_weaklistoffset == 0) {
        /* 父类还没加 weakref 支持（torch < 2.8）⇒ 本缺陷不适用，如实跳过 */
        fprintf(stderr, "[wrfix] %s: tp_weaklistoffset=0，对象不支持 weakref ⇒ 不适用（跳过）\n", tag);
        Py_DECREF(cls);
        return 0;
    }
    if (tp->tp_dealloc == wrapper) {
        fprintf(stderr, "[wrfix] %s: 已是本补丁（幂等跳过）\n", tag);
        *ok_slot = 1;
        Py_DECREF(cls);
        return 0;
    }

    destructor cur = tp->tp_dealloc;
    if (cur == NULL) {
        fprintf(stderr, "[wrfix] %s: tp_dealloc 为空，异常，跳过\n", tag);
        Py_DECREF(cls);
        return 0;
    }
    if (make_writable(&tp->tp_dealloc, sizeof(tp->tp_dealloc)) != 0) {
        /* 只读页放开失败也不放弃：堆类型本来就可写，下面照写 */
        fprintf(stderr, "[wrfix] %s: mprotect 失败（按可写处理）\n", tag);
    }
    *orig_slot = cur;
    tp->tp_dealloc = wrapper;
    *ok_slot = 1;

    fprintf(stderr,
            "[wrfix] %s: 已打补丁 type=%p  orig_dealloc=%p -> %p（先 ClearWeakRefs 再回跳）\n",
            tag, (void *)tp, (void *)cur, (void *)wrapper);
    Py_DECREF(cls);
    return 0;
}

static PyObject *wrfix_info(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    return Py_BuildValue("{s:O,s:O,s:O,s:O}",
                         "stream_patched", stream_ok ? Py_True : Py_False,
                         "event_patched", event_ok ? Py_True : Py_False,
                         "orig_stream", PyLong_FromVoidPtr((void *)orig_stream),
                         "orig_event", PyLong_FromVoidPtr((void *)orig_event));
}

static PyMethodDef Methods[] = {
    {"info", wrfix_info, METH_NOARGS, "返回补丁状态与原 dealloc 地址"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef wrfixmodule = {
    PyModuleDef_HEAD_INIT, "wrfix",
    "PPU Stream/Event 退出期 weakref UAF 运行时等价补丁（止血，待厂商修）",
    -1, Methods
};

PyMODINIT_FUNC PyInit_wrfix(void)
{
    /* PPU 走 torch.cuda 命名空间；类型名与 torch 官方一致 */
    patch_one("torch._C", "_CudaStreamBase", "Stream", wr_stream_dealloc, &orig_stream, &stream_ok);
    patch_one("torch._C", "_CudaEventBase", "Event", wr_event_dealloc, &orig_event, &event_ok);
    return PyModule_Create(&wrfixmodule);
}
