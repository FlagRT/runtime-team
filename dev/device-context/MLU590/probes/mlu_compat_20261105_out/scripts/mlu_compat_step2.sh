#!/usr/bin/env bash
# MLU590 兼容性核对 · Step2 注册层两条路径 + 容器配置盘点（不占卡）
# 用法：ssh Mlu-1 'bash -s' < /tmp/mlu_compat_step2.sh
set -u
TAG='harbor.baai.ac.cn/flagos-runtime/flagos-runtime-cambricon-neuware4.4.3:2.2.0'
C=dc-mlu590-hliu553
WD=/tmp/mlu_compat_20261105
mkdir -p "$WD"

echo "=== SELF_CHECK mlu_compat_step2 2026-11-05 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="
echo '--- our runtime container config ---'
docker inspect "$C" --format 'NetworkMode={{.HostConfig.NetworkMode}} Privileged={{.HostConfig.Privileged}} Image={{.Config.Image}}' 2>&1
docker inspect "$C" --format 'Devices={{json .HostConfig.Devices}}' 2>&1 | head -c 1800; echo
docker inspect "$C" --format 'Binds={{json .HostConfig.Binds}}' 2>&1 | head -c 1200; echo
echo '--- model cache on host ---'; ls -ld /srv/data/hf_cache /srv/data/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B 2>&1 | head -6

cat > "$WD/vendor_path.py" <<'PY'
import sys, torch, torch_mlu
OPS = ["aten::mm", "aten::matmul", "aten::add.Tensor", "aten::mul.Tensor", "aten::silu", "aten::triu"]
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return "dump_failed:%s:%s" % (type(e).__name__, e)
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:2]
print("=== VENDOR PATH (no flag_gems import) ===", flush=True)
print("torch:", torch.__version__, "torch_mlu:", torch_mlu.__version__, flush=True)
print("flag_gems_in_sys_modules:", "flag_gems" in sys.modules, flush=True)
print("triton_in_sys_modules:", "triton" in sys.modules, flush=True)
for op in OPS:
    print("  %-18s %s" % (op, priv(op)), flush=True)
PY

cat > "$WD/gems_path.py" <<'PY'
import sys, torch, torch_mlu
OPS = ["aten::mm", "aten::matmul", "aten::add.Tensor", "aten::mul.Tensor", "aten::silu", "aten::triu"]
def priv(op):
    try:
        t = torch._C._dispatch_dump_table(op)
    except Exception as e:
        return "dump_failed:%s:%s" % (type(e).__name__, e)
    return [l.strip() for l in t.splitlines() if "PrivateUse1" in l][:2]
print("=== GEMS PATH (import + enable) ===", flush=True)
try:
    import triton
    print("triton:", triton.__version__, triton.__file__, flush=True)
except Exception as e:
    print("triton IMPORT_FAIL:", type(e).__name__, e, flush=True)
try:
    import flag_gems
    print("flag_gems:", getattr(flag_gems, "__version__", "no __version__"), flag_gems.__file__, flush=True)
except Exception as e:
    print("flag_gems IMPORT_FAIL:", type(e).__name__, e, flush=True)
    sys.exit(0)
before = {op: priv(op) for op in OPS}
try:
    flag_gems.enable()
    print("ENABLE_OK", flush=True)
except Exception as e:
    print("ENABLE_FAIL:", type(e).__name__, str(e).splitlines()[0][:200], flush=True)
after = {op: priv(op) for op in OPS}
for op in OPS:
    print("  %-18s before=%s" % (op, before[op]), flush=True)
    print("  %-18s after =%s" % (op, after[op]), flush=True)
    print("  %-18s changed=%s" % (op, before[op] != after[op]), flush=True)
PY

echo '--- L1b vendor path (bash -lc, no device) ---'
docker run --rm --network none -v "$WD/vendor_path.py:/vendor_path.py:ro" --entrypoint bash "$TAG" -lc '/flagos/bin/python3 /vendor_path.py' 2>&1 | grep -v -E 'CNDEV|cnrtError|UserWarning|warnings.warn' | head -30
echo '--- L1b gems path (bash -lc, no device) ---'
docker run --rm --network none -v "$WD/gems_path.py:/gems_path.py:ro" --entrypoint bash "$TAG" -lc '/flagos/bin/python3 /gems_path.py' 2>&1 | grep -v -E 'CNDEV|cnrtError|UserWarning|warnings.warn' | head -45
echo "=== END step2 ==="
