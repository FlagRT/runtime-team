#!/usr/bin/env bash
# 壁仞 BR_0：dist 排错 —— 错名为何不立刻报错？spawn 是否根本没跑？
set -u
C=dc-biren-hliu553
echo "=== SELF_CHECK biren_dist_dbg 2026-11-11 ==="

cat > /tmp/dbg1.py <<'PY'
# 单进程、不做 spawn：错名应在 init_process_group 就地报错
import os
import torch
import torch_br  # noqa: F401
import torch.distributed as dist

print("[dbg1] start", flush=True)
for name in ("bccl", "sccl"):
    try:
        dist.init_process_group(backend=name, init_method="tcp://127.0.0.1:29531",
                                world_size=1, rank=0)
        print("[dbg1] %-5s init OK（world=1 单进程）" % name, flush=True)
        dist.destroy_process_group()
    except Exception as e:
        print("[dbg1] %-5s FAIL %s: %s" % (name, type(e).__name__, str(e).splitlines()[0][:110]), flush=True)
print("[dbg1] done", flush=True)
PY
docker cp /tmp/dbg1.py "$C":/tmp/dbg1.py >/dev/null 2>&1
echo '--- 1. 单进程 world=1：错名 vs sccl ---'
timeout 90 docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dbg1.py' \
  2>&1 | grep -avE 'UserWarning|warnings.warn|GeneratorForPrivateuse1' | head -10
echo "rc=$?"

cat > /tmp/dbg2.py <<'PY'
# spawn 是否执行到 worker？给 worker 首行加打印
import os
import torch
import torch.multiprocessing as mp

def worker(rank):
    print("[worker %d] entered" % rank, flush=True)
    import torch_br  # noqa: F401
    print("[worker %d] torch_br imported" % rank, flush=True)
    import torch.distributed as dist
    print("[worker %d] before init" % rank, flush=True)
    dist.init_process_group(backend="sccl", init_method="tcp://127.0.0.1:29541",
                            world_size=2, rank=rank)
    print("[worker %d] init OK" % rank, flush=True)
    dist.destroy_process_group()
    print("[worker %d] done" % rank, flush=True)

if __name__ == "__main__":
    print("[main] spawning", flush=True)
    mp.spawn(worker, args=(), nprocs=2, join=True)
    print("[main] joined", flush=True)
PY
docker cp /tmp/dbg2.py "$C":/tmp/dbg2.py >/dev/null 2>&1
echo
echo '--- 2. spawn 逐步打印（90s 窗口）---'
timeout 90 docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dbg2.py' \
  2>&1 | grep -avE 'UserWarning|warnings.warn|GeneratorForPrivateuse1' | head -14
echo "rc=$?"
docker exec "$C" rm -f /tmp/dbg1.py /tmp/dbg2.py
echo "=== END dbg ==="
