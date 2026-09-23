#!/usr/bin/env python3
"""服务化 vs 前向 吞吐差异的**拆解探针**（P800 阶段 3）。

问题：同一模型、同一张卡、同一批文本（3 句 × 5 轮）
    A 原型单卡前向（transformers，无 vLLM） ≈ 53.12 句/s, p50 56.17 ms
    C vLLM 服务化（HTTP /v1/embeddings）    ≈ 30.70 句/s, p50 96.4 ms
差在哪？本脚本补上中间那一档：

    B vLLM **offline 引擎**（`from vllm import LLM`，同 conda env、同 FL 插件、无 HTTP）

判读规则
    B ≈ A ⇒ 差异主要来自 **HTTP + 服务端调度/序列化**
    B ≈ C ⇒ 差异来自 **vLLM 引擎与插件算子路径**（与前向不是同一套执行栈）
    B 明显低于两者 ⇒ 两者各自有不同开销，需再看引擎内部

单变量：`EAGER`（1 = 关图捕获，与默认服务化一致；0 = 启用图捕获）。

环境要求（与 F2 一致）：`PYTHONPATH=/env/FlagGems/src`、`VLLM_FL_*`、`CUDA_VISIBLE_DEVICES`。
"""
from __future__ import annotations

import json
import os
import time

SAME = ["如何申请退款", "退款流程怎么走", "我要退货"]
MODEL = os.environ.get(
    "DC_MODEL",
    "/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
    "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
EAGER = os.environ.get("EAGER", "1") == "1"
ROUNDS = int(os.environ.get("ROUNDS", "5"))
OUT = os.environ.get("OUT", "/workspace/out_serve/B3_offline_engine_result.json")


def main() -> None:
    res: dict = {"mode": "vllm_offline", "model": MODEL, "enforce_eager": EAGER,
                 "rounds": ROUNDS, "batch": len(SAME)}

    from vllm import LLM

    t0 = time.time()
    llm = LLM(model=MODEL, runner="pooling", convert="embed",
              max_model_len=4096, gpu_memory_utilization=0.25,
              enforce_eager=EAGER)
    res["engine_load_s"] = round(time.time() - t0, 2)
    print(f"[1] 引擎加载 {res['engine_load_s']} s（enforce_eager={EAGER}）", flush=True)

    embed = getattr(llm, "embed", None) or getattr(llm, "encode", None)
    if embed is None:
        res["error"] = "该 vLLM 版本的 LLM 对象既无 embed() 也无 encode()"
        print("✗", res["error"]); _dump(res); return

    for _ in range(2):                      # 预热
        embed(SAME)

    lat = []
    for _ in range(ROUNDS):
        t = time.time()
        outs = embed(SAME)
        lat.append(time.time() - t)

    lat_sorted = sorted(lat)
    avg = sum(lat) / len(lat)
    res["perf"] = {
        "avg_latency_ms": round(avg * 1000, 2),
        "p50_latency_ms": round(lat_sorted[len(lat_sorted) // 2] * 1000, 2),
        "max_latency_ms": round(lat_sorted[-1] * 1000, 2),
        "sent_per_s": round(len(SAME) / avg, 2),
    }
    first = outs[0] if isinstance(outs, list) and outs else None
    vec = getattr(first, "outputs", None) if first is not None else None
    res["sanity"] = {"n_out": len(outs) if isinstance(outs, list) else None,
                     "has_outputs": vec is not None}
    print(f"[2] offline 引擎: {res['perf']['sent_per_s']} 句/s, "
          f"p50 {res['perf']['p50_latency_ms']} ms, avg {res['perf']['avg_latency_ms']} ms")
    _dump(res)


def _dump(res: dict) -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("结果已写入", OUT)


if __name__ == "__main__":
    main()
