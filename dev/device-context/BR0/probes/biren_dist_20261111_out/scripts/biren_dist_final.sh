#!/usr/bin/env bash
# 壁仞 BR_0：集合通信判据 · 收尾（汇总三段证据 + world=2 supa 张量最终复核）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_dist_final_${RUNID}
mkdir -p "$HOSTOUT"
echo "=== SELF_CHECK biren_dist_final $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"

cat > /tmp/d_final.py <<'PY'
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
PY

docker cp /tmp/d_final.py "$C":/tmp/d_final.py >/dev/null 2>&1
timeout 240 docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/d_final.py' \
  > /tmp/dist_final.log 2>&1
RC=$?
echo "rc=$RC"
grep -aE 'SCCL_VERSION|BACKEND_NAME|GOT|EXPECT|MATCH|DONE|Error' /tmp/dist_final.log | head -8

cp /tmp/dist_final.log /tmp/d_final.py "$HOSTOUT"/ 2>/dev/null
docker exec "$C" rm -f /tmp/d_final.py
echo "HOSTOUT=$HOSTOUT"
echo "=== END $RUNID ==="
