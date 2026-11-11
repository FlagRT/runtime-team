import ctypes

torchlib = "/usr/local/lib/python3.10/dist-packages/torch/lib"
brlib = "/usr/local/lib/python3.10/dist-packages/torch_br/lib"

# torch_br 的扩展依赖 torch_python；但进程里其实已加载全部依赖（import torch_br 成功过）
# ⇒ 走“复用已加载句柄”路线：RTLD_NOLOAD 只在已加载时成功，配合 dlinfo 找基址太绕。
# 更简单：让 python 自己先 import（把依赖全带进来），再 dlopen 同一 .so 会直接复用。
import torch
import torch_br  # noqa: F401  此时 libtorch_br.so 已在进程内

lib = ctypes.CDLL(brlib + "/libtorch_br.so")  # 依赖已满足，直接拿句柄

class StdString(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("buf", ctypes.c_char * 16)]

fn = lib._ZN7c10d_br14getScclVersionB5cxx11Ev
fn.restype = ctypes.c_void_p
s = StdString()
fn(ctypes.byref(s))
val = s.buf[:s.size].decode() if s.size < 16 else ctypes.string_at(s.ptr, s.size).decode()
print("getScclVersion() =", repr(val))
