import ctypes
import torch
import torch_br  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp

def sccl_version():
    lib = ctypes.CDLL("/usr/local/lib/python3.10/dist-packages/torch_br/lib/libtorch_br.so")
    class S(ctypes.Structure):
        _fields_ = [("ptr", ctypes.c_void_p), ("size", ctypes.c_size_t), ("buf", ctypes.c_char * 16)]
    fn = lib._ZN7c10d_br14getScclVersionB5cxx11Ev
    fn.restype = ctypes.c_void_p
    s = S()
    fn(ctypes.byref(s))
    return s.buf[:s.size].decode() if s.size < 16 else ctypes.string_at(s.ptr, s.size).decode()

def worker(rank):
    torch_br.supa.set_device(rank)
    dist.init_process_group(backend="sccl", init_method="tcp://127.0.0.1:29581",
                            world_size=2, rank=rank)
    x = (torch.arange(8, dtype=torch.float64) + rank * 8).to(f"supa:{rank}")
    dist.all_reduce(x)
    got = x.cpu()
    exp = torch.arange(8, dtype=torch.float64) * 2 + 8
    if rank == 0:
        print("SCCL_VERSION=", sccl_version(), flush=True)
        print("BACKEND_NAME=sccl (torch_br/__init__.py:179, devices=[supa,cpu])", flush=True)
        print("GOT     =", [round(v, 2) for v in got.tolist()], flush=True)
        print("EXPECT  =", [round(v, 2) for v in exp.tolist()], flush=True)
        print("MATCH   =", bool(torch.allclose(got, exp)), flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("BIREN_DIST_PROBE_DONE", flush=True)
