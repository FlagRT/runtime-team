#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_serve_health.py — P3 服务化 O2 验收：serve 就绪探测 + 预热 + 单请求验证

【用途】等 vllm serve 起来 → 预热（坑 A1）→ 发请求验证输出正确 → 落 JSON 结果
【用法】python3 probe_serve_health.py [--port 8100] [--wait 1800] [--out serve_health_result.json]
【判定】SERVE_READY_PASS：服务就绪 + 预热完成 + 4 条 prompt 全部返回非空且无 NaN

设计纪律：
- 坑 A1：首次请求 attention 极慢，必须先发 1 条短请求预热，再计时正式请求
- 坑 A2：EngineCore 是 spawn 子进程，主进程读不到其 stats，本脚本只做 HTTP 层探测
"""
import argparse
import json
import time
import urllib.error
import urllib.request

PROMPTS = [
    "The capital of France is",
    "量子计算的基本原理是",
    "def fibonacci(n):",
    "人工智能在医疗领域的应用包括",
]


def http_post(port, path, payload, timeout=600):
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get(port, path, timeout=10):
    url = f"http://127.0.0.1:{port}{path}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status


def wait_ready(port, max_wait, log=print):
    """轮询 /health 直到 200 或超时"""
    deadline = time.time() + max_wait
    last_err = ""
    waited = 0
    while time.time() < deadline:
        try:
            if http_get(port, "/health") == 200:
                log(f"[wait] 服务就绪（等待 {waited:.0f}s）")
                return True, waited
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
        time.sleep(5)
        waited += 5
        if waited % 60 == 0:
            log(f"[wait] 已等 {waited}s，仍未就绪（{last_err[:60]}）")
    return False, waited


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--wait", type=int, default=1800, help="就绪等待上限秒（坑 A1）")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="serve_health_result.json")
    args = ap.parse_args()

    result = {
        "verdict": "SERVE_READY_FAIL",
        "port": args.port,
        "timing": {},
        "checks": {},
        "outputs": [],
        "note": "",
    }

    # ── 1. 等待就绪 ──
    print(f"[1/3] 等待服务就绪（上限 {args.wait}s，坑 A1 首次可能很慢）...")
    ok, waited = wait_ready(args.port, args.wait)
    result["timing"]["ready_wait_s"] = round(waited, 1)
    result["checks"]["ready"] = ok
    if not ok:
        result["note"] = f"服务在 {args.wait}s 内未就绪"
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[FAIL] {result['note']}")
        return 1

    # ── 2. 预热（坑 A1）──
    print("[2/3] 预热：发 1 条短请求（首个 attention 可能极慢）...")
    t0 = time.time()
    try:
        http_post(
            args.port,
            "/v1/completions",
            {"model": "qwen3-4b", "prompt": "Hi", "max_tokens": 8, "temperature": 0},
        )
    except Exception as e:  # noqa: BLE001
        result["note"] = f"预热请求失败: {e}"
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[FAIL] {result['note']}")
        return 1
    t_preheat = time.time() - t0
    result["timing"]["preheat_s"] = round(t_preheat, 2)
    print(f"[2/3] 预热完成 {t_preheat:.2f}s")

    # ── 3. 正式请求（greedy，坑 A6 纪律）──
    print(f"[3/3] 正式请求 {len(PROMPTS)} 条（greedy temperature=0）...")
    t1 = time.time()
    texts = []
    total_tokens = 0
    for p in PROMPTS:
        r = http_post(
            args.port,
            "/v1/completions",
            {
                "model": "qwen3-4b",
                "prompt": p,
                "max_tokens": args.max_tokens,
                "temperature": 0,
            },
        )
        txt = r["choices"][0]["text"]
        tokens = r.get("usage", {}).get("completion_tokens", 0)
        total_tokens += tokens
        texts.append(txt)
        result["outputs"].append(
            {"prompt": p, "text": txt[:200], "completion_tokens": tokens}
        )
    t_infer = time.time() - t1
    result["timing"]["infer_s"] = round(t_infer, 2)
    result["timing"]["throughput_tok_s"] = (
        round(total_tokens / t_infer, 1) if t_infer > 0 else 0
    )
    result["timing"]["total_tokens"] = total_tokens

    ok = all(len(t.strip()) > 0 for t in texts) and not any(
        "nan" in t.lower() for t in texts
    )
    result["checks"]["outputs_non_empty"] = all(len(t.strip()) > 0 for t in texts)
    result["checks"]["no_nan"] = not any("nan" in t.lower() for t in texts)
    result["verdict"] = "SERVE_READY_PASS" if ok else "SERVE_READY_FAIL"

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 判定：{result['verdict']} ===")
    print(f"就绪等待 {result['timing']['ready_wait_s']}s | 预热 {result['timing']['preheat_s']}s")
    print(
        f"推理 {result['timing']['infer_s']}s | {result['timing']['throughput_tok_s']} tok/s"
    )
    for o in result["outputs"]:
        print(f"  · {o['prompt'][:28]!r} → {o['text'][:60]!r}")
    print(f"结果：{args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
