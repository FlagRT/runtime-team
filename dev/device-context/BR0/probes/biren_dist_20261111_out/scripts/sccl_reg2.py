import torch_br
import torch.distributed as dist

sub = torch_br._C._distributed_c10d
# dir() 已确认只有 ProcessGroupSCCL / SCCLOptions 对外；版本符号 getScclVersion 在 libtorch_br.so
# （nm -D 可见 c10d_br::getScclVersion）但未通过 pybind 暴露 ⇒ Python 面读不到版本号。
# 尝试通过 C 符号直呼（ctypes）——找不到导出路径则如实登记「版本号不可达」。
import ctypes
so = "/usr/local/lib/python3.10/dist-packages/torch_br/lib/libtorch_br.so"
try:
    lib = ctypes.CDLL(so)
    fn = getattr(lib, "_ZN7c10d_br14getScclVersionB5cxx11Ev", None)
    print("ctypes symbol found:", fn is not None)
except Exception as e:
    print("ctypes fail:", type(e).__name__, e)

# 注册表口径：Backend 映射里 sccl 的形态
print("is_backend_available(sccl):", dist.is_backend_available("sccl"))
b = dist.Backend.SCCL if hasattr(dist.Backend, "SCCL") else None
print("Backend.SCCL attr:", b)
# devices 声明（torch_br 注册时写了 devices=["supa","cpu"]）
import torch
print("Backend 除 sccl 外有 SUPA 专属项:", [x for x in dir(dist.Backend) if "SUPA" in x.upper() or "SCCL" in x.upper()])
