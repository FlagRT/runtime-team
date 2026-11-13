#!/usr/bin/env bash
# 壁仞 BR_0：集合通信判据 · 第三步（修正版 worker：不在进程内改可见卡变量）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_dist3_${RUNID}
mkdir -p "$HOSTOUT"
echo "=== SELF_CHECK biren_dist_fn3 $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"
echo "--- 卡况 ---"
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo

cat > /tmp/dist_worker3.py <<'PY'
import os
import torch
import torch_br  # noqa: F401  触发 sccl 注册（devices=["supa","cpu"]）
import torch.distributed as dist
import torch.multiprocessing as mp

PORT = os.environ.get("DIST_PORT", "29551")

def worker(rank):
    # ⚠️ 不在 worker 内改 BIREN_VISIBLE_DEVICES（容器级注入，进程内改会挂住设备上下文）
    torch_br.supa.set_device(rank)
    dist.init_process_group(backend="sccl", init_method=f"tcp://127.0.0.1:{PORT}",
                            world_size=2, rank=rank)
    x = (torch.arange(8, dtype=torch.float64) + rank * 8).to(f"supa:{rank}")
    dist.all_reduce(x)
    got = x.cpu()
    # 期望：各 rank i+8r 逐元素求和 = i*2 + 8*(0+1) = 2i+8
    exp = torch.arange(8, dtype=torch.float64) * 2 + 8
    if rank == 0:
        ok = bool(torch.allclose(got, exp))
        print("GOT   =", [round(v, 2) for v in got.tolist()], flush=True)
        print("EXPECT=", [round(v, 2) for v in exp.tolist()], flush=True)
        print("ALLREDUCE_OK=", ok, flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    print("BACKEND=sccl devices=[supa,cpu]（torch_br/__init__.py:179 注册）", flush=True)
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("BIREN_DIST_ALLREDUCE_PASS", flush=True)
PY

docker cp /tmp/dist_worker3.py "$C":/tmp/dist_worker3.py >/dev/null 2>&1

echo '--- A. sccl 2-rank all_reduce（双卡）---'
timeout 240 docker exec -e PYTHONDONTWRITEBYTECODE=1 -e DIST_PORT=29551 "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dist_worker3.py' \
  > /tmp/dist3_sccl.log 2>&1
RC_A=$?
echo "rc_A=$RC_A"
grep -aE 'BACKEND|GOT|EXPECT|ALLREDUCE|PASS|Error|Traceback' /tmp/dist3_sccl.log | head -8

echo
echo '--- B. 带张量搬运的完整链路（CPU 张量 all_reduce，验 sccl 的 cpu 设备面）---'
cat > /tmp/dist_cpu.py <<'PY'
import os
import torch
import torch_br  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp

def worker(rank):
    dist.init_process_group(backend="sccl", init_method="tcp://127.0.0.1:29561",
                            world_size=2, rank=rank)
    x = torch.arange(4, dtype=torch.float64) + rank
    dist.all_reduce(x)
    if rank == 0:
        exp = torch.tensor([1.0, 3.0, 5.0, 7.0])
        print("CPU_GOT=", x.tolist(), "EXPECT=", exp.tolist(), "OK=", bool(torch.allclose(x, exp)), flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("BIREN_DIST_CPU_ALLREDUCE_PASS", flush=True)
PY
docker cp /tmp/dist_cpu.py "$C":/tmp/dist_cpu.py >/dev/null 2>&1
timeout 180 docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dist_cpu.py' \
  > /tmp/dist3_cpu.log 2>&1
RC_B=$?
echo "rc_B=$RC_B"
grep -aE 'CPU_GOT|PASS|Error|Traceback' /tmp/dist3_cpu.log | head -5

cp /tmp/dist3_sccl.log /tmp/dist3_cpu.log /tmp/dist_worker3.py /tmp/dist_cpu.py "$HOSTOUT"/ 2>/dev/null
docker exec "$C" rm -f /tmp/dist_worker3.py /tmp/dist_cpu.py
echo "HOSTOUT=$HOSTOUT"
echo "=== END $RUNID ==="
