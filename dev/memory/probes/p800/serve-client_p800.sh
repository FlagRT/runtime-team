#!/usr/bin/env bash
# S3 vllm serve 请求测试（Route A / P800）
# 用法: bash routeA_s3_serve_client.sh <port> <模型名>
set -e
PORT=${1:-8001}
MODEL=${2:-Qwen3-4B}
BASE="http://127.0.0.1:${PORT}/v1"
echo "=== 1) chat completion (计时) ==="
python3 - "$BASE" "$MODEL" << 'EOF'
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "用一句话介绍量子计算。"}],
    "max_tokens": 64,
    "temperature": 0.0,
}).encode()
req = urllib.request.Request(base + "/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.loads(r.read())
dt = time.time() - t0
msg = resp["choices"][0]["message"]["content"]
usage = resp.get("usage", {})
n_out = usage.get("completion_tokens", 0)
print(f"TTFT+生成总耗时: {dt:.2f}s, 输出 {n_out} tokens, {n_out/dt:.1f} tok/s")
print(f"回复: {msg!r}")
print("usage:", usage)
EOF
echo
echo "=== 2) /v1/models ==="
curl -s "${BASE}/models" | head -c 300
echo
echo "=== 3) 第二次请求 (已预热) ==="
python3 - "$BASE" "$MODEL" << 'EOF'
import json, sys, time, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "1+1=?"}],
    "max_tokens": 16,
}).encode()
req = urllib.request.Request(base + "/chat/completions", body, {"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.loads(r.read())
dt = time.time() - t0
print(f"耗时: {dt:.2f}s, 回复: {resp['choices'][0]['message']['content']!r}")
EOF
