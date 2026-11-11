#!/usr/bin/env bash
# 壁仞 BR_0 · 镜像内容探查（docker run --rm，不挂设备、只读）
# 用法：ssh Biren-1 'bash -s' < 本脚本
set -u
echo "=== SELF_CHECK biren_image_probe 2026-11-11 ==="
echo "=== [HOST] $(hostname) $(date '+%F %T %Z') ==="

cat > /tmp/img_probe.py <<'PY'
import sys
print("=== IMAGE SELF-CHECK ===")
print("os-release:", [l for l in open("/etc/os-release").read().splitlines() if l.startswith(("PRETTY_NAME",))])
print("python:", sys.version.split()[0], sys.executable)
def tryimp(name):
    try:
        m = __import__(name)
        return "OK " + str(getattr(m, "__version__", "no __version__"))
    except Exception as e:
        return "FAIL %s: %s" % (type(e).__name__, str(e).splitlines()[0][:120])
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
for n in ["supa", "torch_supa", "torch_biren", "biren", "birensupa", "br_torch"]:
    r = tryimp(n)
    print("import %-12s -> %s" % (n, r))
    if r.startswith("OK"):
        print("   file:", __import__(n).__file__)
# 设备命名空间判别
for ns in ["biren", "supa", "cuda", "xpu", "npu", "mlu"]:
    if hasattr(torch, ns):
        mod = getattr(torch, ns)
        cnt = "device_count=%s" % mod.device_count() if hasattr(mod, "device_count") else "?"
        print("torch.%s: exists, %s" % (ns, cnt))
PY

for IMG in \
  'birensupa-pytorch:26.05.rc1-py310-pt2.8.0-br1xx' \
  'biren-kernel-dev:torch-2.8.0-sdk-1.12.0.2-20261107' ; do
  echo ""
  echo "############ IMAGE: $IMG ############"
  docker run --rm -v /tmp/img_probe.py:/img_probe.py:ro --entrypoint python3 "$IMG" /img_probe.py 2>&1 | grep -vE '^\s*$' | head -25
done
rm -f /tmp/img_probe.py
echo "=== END image probe ==="
