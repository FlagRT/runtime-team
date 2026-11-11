#!/usr/bin/env bash
# 壁仞 BR_0：集合通信后端名判据 · 第二步真 2-rank 功能验证
# A. sccl 后端 all_reduce（BIREN_DIST_ALLREDUCE）
# B. 错名必须报错（不静默回落 gloo）（BIREN_DIST_NO_SILENT_FALLBACK）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_dist_${RUNID}
mkdir -p "$HOSTOUT"
echo "=== SELF_CHECK biren_dist_fn $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"
echo '--- 卡况 ---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo

cat > /tmp/dist_worker.py <<'PY'
import os, sys
import torch
import torch_br  # noqa: F401
import torch.distributed as dist
import torch.multiprocessing as mp

BACKEND = os.environ.get("DIST_BACKEND", "sccl")
MODE = os.environ.get("DIST_MODE", "allreduce")

def worker(rank, world):
    os.environ["BIREN_VISIBLE_DEVICES"] = "0,1"
    torch_br.supa.set_device(rank)
    init = dist.init_process_group(
        backend=BACKEND,
        init_method=f"tcp://127.0.0.1:295{abs(hash(BACKEND+MODE)) % 90 + 10}",
        world_size=world, rank=rank,
    )
    t = torch.arange(8, dtype=torch.float64) + rank * 8
    t = t.to(f"supa:{rank}")
    dist.all_reduce(t)
    t2 = t.cpu()
    exp = sum(float(i % world) * 8 + i for i in range(8)) if MODE == "allreduce" else None
    if rank == 0:
        ok = exp is not None and abs(float(t2.sum()) - sum(float(v) for v in [sum(range(8)) + r * 64 for r in range(world)])) < 1e-6
        print("ALLREDUCE_VALUE=", [round(float(v), 2) for v in t2.tolist()], flush=True)
        print("ALLREDUCE_SUM=", round(float(t2.sum()), 2), "EXPECT=", round(sum(sum(range(8)) + r * 64 for r in range(world)), 2), flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
    if MODE == "allreduce":
        print("BIREN_DIST_ALLREDUCE_PASS", flush=True)
PY

docker cp /tmp/dist_worker.py "$C":/tmp/dist_worker.py >/dev/null 2>&1

echo '--- A. sccl all_reduce（2 rank × 双卡）---'
timeout 180 docker exec -e PYTHONDONTWRITEBYTECODE=1 -e DIST_BACKEND=sccl -e DIST_MODE=allreduce "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dist_worker.py' \
  > /tmp/dist_sccl.log 2>&1
RC_A=$?
echo "rc_A=$RC_A"
grep -aE 'ALLREDUCE|PASS|Error|error|Traceback|assert' /tmp/dist_sccl.log | head -8

echo
echo '--- B. 错名必须报错（给 "bccl"）---'
timeout 60 docker exec -e PYTHONDONTWRITEBYTECODE=1 -e DIST_BACKEND=bccl -e DIST_MODE=badname "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dist_worker.py' \
  > /tmp/dist_bad.log 2>&1
RC_B=$?
echo "rc_B=$RC_B（非 0 = 如实报错，好）"
grep -aE 'Error|error|Traceback|Invalid|backend' /tmp/dist_bad.log | head -5

echo
echo '--- C. 静默回落对照：不指定后端名跑同一脚本会怎样（gloo 兜底检查）---'
timeout 180 docker exec -e PYTHONDONTWRITEBYTECODE=1 -e DIST_BACKEND=gloo -e DIST_MODE=allreduce "$C" bash -c \
  '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/dist_worker.py' \
  > /tmp/dist_gloo.log 2>&1
RC_C=$?
echo "rc_C=$RC_C"
grep -aE 'ALLREDUCE|PASS|Error|error' /tmp/dist_gloo.log | head -6

cp /tmp/dist_sccl.log /tmp/dist_bad.log /tmp/dist_gloo.log "$HOSTOUT"/ 2>/dev/null
cp /tmp/dist_worker.py "$HOSTOUT"/ 2>/dev/null
docker exec "$C" rm -f /tmp/dist_worker.py
echo "HOSTOUT=$HOSTOUT"
echo "=== END $RUNID ==="
