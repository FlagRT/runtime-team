#!/usr/bin/env bash
# 壁仞 BR_0：集合通信判据 · 第四步 补充证据
# A2. sccl CPU 张量行为定性（devices 含 "cpu" 但 all_reduce 拒 CPU 张量 ⇒ 实用面=supa-only）
# A3. is_available 条件与 SCCL 库实体（自证注册链）
set -u
C=dc-biren-hliu553
RUNID=$(date +%Y%m%d_%H%M%S)
HOSTOUT=/tmp/biren_dist4_${RUNID}
mkdir -p "$HOSTOUT"
echo "=== SELF_CHECK biren_dist4 $RUNID ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"

echo '--- A2. supa 张量单卡 world=1 + CPU 张量行为 ---'
cat > /tmp/d4.py <<'PY'
import os, sys
import torch
import torch_br  # noqa: F401
import torch.distributed as dist

mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
dist.init_process_group(backend="sccl", init_method="tcp://127.0.0.1:29571",
                        world_size=1, rank=0)
if mode == "supa":
    x = (torch.arange(8, dtype=torch.float64)).to("supa:0")
    dist.all_reduce(x)
    print("SUPA_W1_OK got=", [round(v, 2) for v in x.cpu().tolist()], flush=True)
elif mode == "cpu":
    try:
        x = torch.arange(8, dtype=torch.float64)
        dist.all_reduce(x)
        print("CPU_OK got=", [round(v, 2) for v in x.tolist()], flush=True)
    except Exception as e:
        print("CPU_FAIL %s: %s" % (type(e).__name__, str(e).splitlines()[0][:100]), flush=True)
dist.destroy_process_group()
print("DONE", flush=True)
PY
docker cp /tmp/d4.py "$C":/tmp/d4.py >/dev/null 2>&1
for m in supa cpu; do
  echo "  [mode=$m]"
  timeout 120 docker exec -e PYTHONDONTWRITEBYTECODE=1 "$C" bash -c \
    ". /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 -u /tmp/d4.py $m" \
    2>&1 | grep -aE 'SUPA_W1_OK|CPU_OK|CPU_FAIL|DONE|Error' | head -3
done

echo
echo '--- A3. is_available 条件 + SCCL 库实体 ---'
docker exec "$C" bash -c '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1;
echo "[i] is_available 条件源码（torch_br/__init__.py 注册段之前）:";
grep -n -B2 -A6 "def is_available" /usr/local/lib/python3.10/dist-packages/torch_br/__init__.py | head -20;
echo;
echo "[ii] _distributed_c10d 扩展实体:";
python3 -c "import torch_br._C._distributed_c10d as m; print(m.__file__)" 2>/dev/null;
ls -la /usr/local/lib/python3.10/dist-packages/torch_br/_C/ 2>/dev/null | grep -iE "distrib|sccl" | head -5;
echo;
echo "[iii] 容器内 sccl 相关库:";
find /usr/local/birensupa /usr/local/lib/python3.10/dist-packages/torch_br -name "*sccl*" 2>/dev/null | head -8' 2>&1 | grep -avE 'UserWarning|warnings.warn' | head -30

cp /tmp/d4.py "$HOSTOUT"/ 2>/dev/null
docker exec "$C" rm -f /tmp/d4.py
echo "HOSTOUT=$HOSTOUT"
echo "=== END $RUNID ==="
