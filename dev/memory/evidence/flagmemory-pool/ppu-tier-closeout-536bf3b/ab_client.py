#!/usr/bin/env python3
# ab_client.py -- PPU connector A/B 客户端（2026-11-06 收尾证据）
# 20 条固定中文短 prompt（greedy, max_tokens=64），可选 --recall 追加 4 条
# 共享前缀 prompt 的首放 + 重放（验证 [PpuOffload-SCHED]/load DMA）。
import argparse, json, time, urllib.request, threading, queue, sys

PROMPTS = [
    "用一句话解释什么是张量并行。",
    "用一句话解释什么是流水线并行。",
    "用一句话解释什么是数据并行。",
    "用一句话解释什么是专家并行。",
    "简述KV缓存的作用。",
    "简述页注意力机制的思想。",
    "混合精度训练有什么好处？",
    "什么是梯度检查点？",
    "什么是学习率预热？",
    "什么是余弦退火？",
    "什么是零冗余优化器？",
    "什么是上下文并行？",
    "什么是序列并行？",
    "什么是写时复制？",
    "什么是请求抢占？",
    "什么是块表？",
    "什么是前缀缓存？",
    "什么是激活值卸载？",
    "什么是优化器状态卸载？",
    "什么是动态损失缩放？",
]

PREFIX = ("我们团队研发一套异构计算运行时软件，核心组件包括设备抽象层、"
          "任务调度引擎、显存池管理器与监控诊断模块，目标是让大模型推理"
          "与训练在多种国产加速器上高效运行。")
RECALL = [PREFIX + f"请围绕上述背景，用约五十字介绍其中「{t}」的职责。" for t in
          ("任务调度引擎", "显存池管理器", "设备抽象层", "监控诊断模块")]


def post(url, prompt, max_tokens=64, timeout=900):
    body = json.dumps({"model": "tiny", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return {"latency": round(time.time() - t0, 3), "out": d["choices"][0]["message"]["content"],
            "usage": d.get("usage", {})}


def run_batch(url, tag, prompts, conc):
    results = [None] * len(prompts)
    q = queue.Queue()
    for i, p in enumerate(prompts):
        q.put(i)
    lock = threading.Lock()

    def worker():
        while True:
            try:
                i = q.get_nowait()
            except queue.Empty:
                return
            try:
                results[i] = {"ok": True, **post(url, prompts[i])}
            except Exception as e:
                results[i] = {"ok": False, "err": repr(e)[:200]}
            q.task_done()

    ts = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = round(time.time() - t0, 3)
    ok = [r for r in results if r and r.get("ok")]
    toks = sum(r["usage"].get("completion_tokens", 0) for r in ok)
    print(f"[{tag}] wall={wall}s ok={len(ok)}/{len(prompts)} "
          f"completion_tokens={toks} throughput={toks/wall:.2f} tok/s", flush=True)
    lats = sorted(r["latency"] for r in ok)
    if lats:
        print(f"[{tag}] latency s: min={lats[0]} p50={lats[len(lats)//2]} max={lats[-1]} "
              f"mean={sum(lats)/len(lats):.3f}", flush=True)
    return {"tag": tag, "wall": wall, "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--recall", action="store_true")
    a = ap.parse_args()

    # 健康等待
    for _ in range(120):
        try:
            urllib.request.urlopen(a.url + "/v1/models", timeout=5)
            break
        except Exception:
            time.sleep(5)

    report = {"started": time.strftime("%F %T")}
    report["main"] = run_batch(a.url, "main20", PROMPTS, a.concurrency)
    if a.recall:
        time.sleep(3)
        report["recall_first"] = run_batch(a.url, "recall_first", RECALL, a.concurrency)
        time.sleep(3)
        report["recall_replay"] = run_batch(a.url, "recall_replay", RECALL, a.concurrency)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("saved:", a.out, flush=True)


if __name__ == "__main__":
    main()
