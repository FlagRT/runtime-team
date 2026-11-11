#!/usr/bin/env bash
set -u
echo "=== SELF_CHECK biren_step4 2026-11-11 ==="
echo '--- docker data-root ---'; echo 'G7kL!3pQ9sX' | sudo -S -p '' docker info 2>/dev/null | grep -E 'Data Root|Storage Driver|Registry'
echo '--- docker images (biren/flagos/pytorch) ---'; echo 'G7kL!3pQ9sX' | sudo -S -p '' docker images --format '{{.Repository}}:{{.Tag}}|{{.Size}}|{{.CreatedSince}}' 2>/dev/null | grep -iE 'biren|br|flagos|pytorch|vllm|sglang' | head -20
echo '--- /data/biren-sdk ---'; ls /data/biren-sdk/ 2>/dev/null | head -10
echo '--- /data/hf_cache ---'; ls /data/hf_cache/ 2>/dev/null | head -8; ls /data/hf_cache/hub 2>/dev/null | head -8
echo '--- /data/docker ---'; ls /data/docker/ 2>/dev/null | head -6
echo '--- readme.txt (share package) ---'; head -30 /share/package/readme.txt 2>/dev/null
echo "=== END step4 ==="
