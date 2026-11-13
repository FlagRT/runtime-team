#!/usr/bin/env bash
# 壁仞 BR_0：集合通信后端名探测 · 第一步注册层侦察（只读，不动卡）
set -u
C=dc-biren-hliu553
echo "=== SELF_CHECK biren_dist_recon 2026-11-11 ==="
echo "[HOST] $(hostname) $(date '+%F %T %Z')"

echo '--- 0. 卡况（先普查再动卡）---'
brsmi gpu --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' '; echo

cat > /tmp/dist_recon.py <<'PY'
import os, sys
import torch
import torch_br  # noqa: F401  必须先 import 才会注册厂商后端
import torch.distributed as dist

print("torch:", torch.__version__, "| torch_br:", getattr(torch_br, "__version__", "?"))
print("torch_br path:", os.path.dirname(torch_br.__file__))
print("dist.is_available:", dist.is_available())
print("gloo:", dist.is_gloo_available(), "| nccl:", dist.is_nccl_available(),
      "| mpi:", dist.is_mpi_available(), "| ucc:", dist.is_ucc_available())

# 触碰设备（纪律：拼厂商字符串前先经设备侧触碰）
print("supa device_count:", torch.supa.device_count())

# 候选后端名逐个探测（is_backend_available 只查注册表，不真正建组）
cands = ["gloo", "nccl", "hccl", "cncl", "bccl", "bcl", "supa", "biren", "br",
         "rccl", "accl", "pccl", "xccl", "xpu", "mlu", "npu"]
print("\n--- is_backend_available（注册表口径）---")
for n in cands:
    try:
        print("  %-8s %s" % (n, dist.is_backend_available(n)))
    except Exception as e:
        print("  %-8s ERR:%s %s" % (n, type(e).__name__, str(e)[:60]))
PY

docker cp /tmp/dist_recon.py "$C":/tmp/dist_recon.py >/dev/null 2>&1
docker exec "$C" bash -c '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 /tmp/dist_recon.py' 2>&1 | grep -viE '^\s*warn|UserWarning|warnings.warn' | head -30

echo
echo '--- 1. torch_br 里 register_backend 的权威定义 ---'
docker exec "$C" bash -c '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1;
TB=$(python3 -c "import torch_br,os;print(os.path.dirname(torch_br.__file__))");
echo "TB=$TB";
grep -rn "register_backend" "$TB" 2>/dev/null | head -10' 2>&1 | head -14

echo
echo '--- 2. 厂商集合通信库实体（libbccl/libbcl/libnccl...）---'
docker exec "$C" bash -c 'find /usr/local/birensupa /usr/lib /usr/local/lib /opt -maxdepth 6 \( -name "*bccl*" -o -name "*bcl*" -o -name "libnccl*" -o -name "*collective*" \) 2>/dev/null | head -12' 2>&1 | head -14

echo
echo '--- 3. vLLM 侧线索（镜像自带 vllm 0.21.0，TP 需要集合通信）---'
docker exec "$C" bash -c '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1;
V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null);
echo "VLLM=$V";
[ -n "$V" ] && grep -rn -iE "supa|biren" "$V/distributed/" 2>/dev/null | grep -viE "\.pyc" | head -12' 2>&1 | head -16
docker exec "$C" rm -f /tmp/dist_recon.py
echo "=== END recon ==="
