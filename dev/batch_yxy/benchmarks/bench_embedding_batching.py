#!/usr/bin/env python
"""A/B/C 对照基准：Qwen3-Embedding 组批策略（实现文档第 7 节对照测试）。

模式：
  A baseline      原路径：整组一次 executor.embed 直通；
  B dynamic       动态组批，单桶（无长度分桶）；
  C bucketed      动态组批 + 长度分桶（可开相邻桶合并/比例约束）。

执行器：
  --executor fake  离线假执行器（确定性向量 + Padding 感知成本模型），可复现；
  --executor vllm  真实 vLLM LLM.embed（需 --model 指向本地权重，NPU 环境）。

同一输入集、同一执行器、同一进程内顺序执行 A/B/C；正确性以 A 为基线比较
（fake 模式应逐元素相等；vllm 模式报告最大绝对差与最小余弦相似度）。
结果打印并写入 benchmarks/results/*.json。

示例：
  .venv/bin/python benchmarks/bench_embedding_batching.py \
      --executor fake --scenario mixed --num 200
  .venv/bin/python benchmarks/bench_embedding_batching.py \
      --executor vllm --model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
      --scenario mixed --num 128 --max-input-tokens 2048
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dynamic_batching import (
    BatchPolicy,
    DynamicBatchEmbedder,
    FakeEmbedExecutor,
    summarize_records,
)
from dynamic_batching.vllm_embed_adapter import VllmEmbedExecutor

sys.path.remove(str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executor", choices=("fake", "vllm"), default="fake")
    p.add_argument("--model", default="/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
    p.add_argument("--scenario", default="mixed",
                   choices=("short", "long", "mixed", "burst", "lowtraffic"))
    p.add_argument("--num", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--burst-size", type=int, default=32)
    p.add_argument("--burst-gap-ms", type=float, default=200.0)
    p.add_argument("--trickle-ms", type=float, default=30.0)
    p.add_argument("--max-batch-size", type=int, default=32)
    p.add_argument("--max-total-tokens", type=int, default=32768)
    p.add_argument("--max-wait-ms", type=float, default=50.0)
    p.add_argument("--bounds", default="64,128,256,512,1024,2048",
                   help="C 模式桶上界，逗号分隔；不足 max-input-tokens 时自动追加")
    p.add_argument("--adjacent-merge", action="store_true")
    p.add_argument("--max-padding-ratio", type=float, default=None)
    p.add_argument("--warmup", type=int, default=1,
                   help=">0 时先用 baseline 跑一遍全部输入做全形状预热（消除编译偏置）")
    p.add_argument("--enable-prefix-caching", action="store_true",
                   help="vLLM 引擎开启 prefix caching（默认关闭：vllm-ascend 0.20.2rc1 "
                        "pooling + aclgraph + prefix cache 组合存在非确定错位，见接入说明）")
    p.add_argument("--enforce-eager", action="store_true",
                   help="关闭图编译（该 build 上图模式 pooling 数值污染更严重）")
    p.add_argument("--fake-fixed-ms", type=float, default=10.0)
    p.add_argument("--fake-per-token-ms", type=float, default=0.002)
    p.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    return p.parse_args()


def build_items(args: argparse.Namespace) -> list[tuple[str, str]]:
    rng = np.random.default_rng(args.seed)
    words = [f"w{j}" for j in range(1024)]
    items = []
    for i in range(args.num):
        if args.scenario == "short":
            length = rng.integers(16, 65)
        elif args.scenario == "long":
            length = rng.integers(512, args.max_input_tokens + 1)
        elif args.scenario == "mixed":
            length = (
                rng.integers(16, 65)
                if rng.random() < 0.6
                else rng.integers(512, args.max_input_tokens + 1)
            )
        else:
            length = rng.integers(16, args.max_input_tokens + 1)
        length = int(min(length, args.max_input_tokens))
        start = (i * 37) % 1000
        text = " ".join(words[(start + j) % 1024] for j in range(length))
        items.append((f"req-{i:05d}", text))
    return items


def token_count_of(executor, text: str) -> int:
    counter = getattr(executor, "token_count", None)
    if counter is not None:
        return int(counter(text))
    return len(text.split())


def finalize_items(args: argparse.Namespace, executor, items):
    trimmed = []
    for rid, text in items:
        while token_count_of(executor, text) > args.max_input_tokens:
            words = text.split()
            text = " ".join(words[: -max(1, len(words) // 8)])
        trimmed.append((rid, text))
    return trimmed


def build_groups(args: argparse.Namespace, items: list[tuple[str, str]]):
    if args.scenario == "burst":
        return [
            items[i : i + args.burst_size]
            for i in range(0, len(items), args.burst_size)
        ], args.burst_gap_ms
    if args.scenario == "lowtraffic":
        return [[item] for item in items], args.trickle_ms
    return [items], 0.0


def build_policies(args: argparse.Namespace):
    bounds = sorted({int(x) for x in args.bounds.split(",") if x.strip()})
    if bounds[-1] < args.max_input_tokens:
        bounds.append(args.max_input_tokens)
    common = dict(
        model_id="Qwen3-Embedding-0.6B",
        max_batch_size=args.max_batch_size,
        max_total_tokens=args.max_total_tokens,
        max_wait_ms=args.max_wait_ms,
        allow_adjacent_bucket_merge=args.adjacent_merge,
        max_padding_ratio=args.max_padding_ratio,
    )
    policy_b = BatchPolicy(
        policy_version="B-nobucket-v1",
        bucket_upper_bounds=(args.max_input_tokens,),
        **common,
    )
    policy_c = BatchPolicy(
        policy_version="C-bucket-v1",
        bucket_upper_bounds=tuple(bounds),
        **common,
    )
    return policy_b, policy_c


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(pct / 100 * (len(ordered) - 1)))))
    return ordered[idx]


def run_mode(mode: str, executor, policy, groups, gap_ms: float):
    timings = {}
    records = []
    batch_exec_ms = {}
    vectors = {}
    wall_start = time.monotonic_ns()
    for idx, group in enumerate(groups):
        if idx and gap_ms > 0:
            time.sleep(gap_ms / 1000.0)
        embedder = DynamicBatchEmbedder(
            executor,
            policy if mode != "A" else None,
            model_version="0.6b",
            execution_profile="bench",
        )
        result = embedder.run(group)
        timings.update(result.timings)
        vectors.update(result.vectors)
        records.extend(result.dispatch_records)
        batch_exec_ms.update(result.batch_exec_ms)
    wall_ms = (time.monotonic_ns() - wall_start) / 1e6
    e2e = [t.e2e_ms for t in timings.values()]
    queue = [t.queue_ms for t in timings.values()]
    return {
        "mode": mode,
        "wall_ms": wall_ms,
        "requests_per_s": len(timings) / (wall_ms / 1000.0),
        "e2e_ms": {
            "p50": percentile(e2e, 50),
            "p95": percentile(e2e, 95),
            "p99": percentile(e2e, 99),
            "max": max(e2e) if e2e else None,
        },
        "queue_ms": {
            "p50": percentile(queue, 50),
            "p95": percentile(queue, 95),
            "max": max(queue) if queue else None,
        },
        "batch_summary": summarize_records(records),
        "batch_exec_ms": batch_exec_ms,
        "executor_calls": getattr(executor, "call_count", None),
        "timings": timings,
        "vectors": vectors,
    }


def compare_with_baseline(baseline_vectors, run_vectors):
    common = sorted(set(baseline_vectors) & set(run_vectors))
    if len(common) != len(baseline_vectors):
        return {"error": "请求 ID 集合不一致", "matched": len(common)}
    max_abs = 0.0
    min_cos = 1.0
    for rid in common:
        a = baseline_vectors[rid].astype(np.float64)
        b = run_vectors[rid].astype(np.float64)
        max_abs = max(max_abs, float(np.max(np.abs(a - b))))
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cos = float(np.dot(a, b) / denom) if denom > 0 else 1.0
        min_cos = min(min_cos, cos)
    return {"max_abs_diff": max_abs, "min_cosine": min_cos, "matched": len(common)}


def main() -> int:
    args = parse_args()
    if args.executor == "fake":
        executor = FakeEmbedExecutor(
            fixed_cost_ms=args.fake_fixed_ms,
            per_token_ms=args.fake_per_token_ms,
            padding_aware=True,
        )
    else:
        from vllm import LLM

        llm = LLM(
            model=args.model,
            max_model_len=max(args.max_input_tokens, 4096),
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
        )
        executor = VllmEmbedExecutor(llm)

    items = finalize_items(args, executor, build_items(args))
    groups, gap_ms = build_groups(args, items)
    policy_b, policy_c = build_policies(args)

    if args.warmup:
        DynamicBatchEmbedder(executor).run(items)
    executor.call_count = 0

    runs = [
        run_mode("A", executor, None, groups, gap_ms),
        run_mode("B", executor, policy_b, groups, gap_ms),
        run_mode("C", executor, policy_c, groups, gap_ms),
    ]

    length_dist = Counter()
    for _, text in items:
        n = token_count_of(executor, text)
        for bound in policy_c.bucket_upper_bounds:
            if n <= bound:
                length_dist[f"<={bound}"] += 1
                break

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "executor": args.executor,
        "model": args.model if args.executor == "vllm" else "fake",
        "engine": (
            {
                "max_model_len": max(args.max_input_tokens, 4096),
                "enable_prefix_caching": args.enable_prefix_caching,
            }
            if args.executor == "vllm"
            else None
        ),
        "scenario": args.scenario,
        "num_requests": len(items),
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "policy_C": {
            "bucket_upper_bounds": policy_c.bucket_upper_bounds,
            "max_batch_size": policy_c.max_batch_size,
            "max_total_tokens": policy_c.max_total_tokens,
            "max_wait_ms": policy_c.max_wait_ms,
            "allow_adjacent_bucket_merge": policy_c.allow_adjacent_bucket_merge,
            "max_padding_ratio": policy_c.max_padding_ratio,
        },
        "length_distribution": dict(length_dist),
        "runs": [],
        "correctness": {},
    }
    for run in runs:
        report["runs"].append(
            {k: v for k, v in run.items() if k not in ("timings", "vectors")}
        )
        if run["mode"] != "A":
            report["correctness"][run["mode"]] = compare_with_baseline(
                runs[0]["vectors"], run["vectors"]
            )

    print(f"\n=== A/B/C 对照（executor={args.executor}, scenario={args.scenario}, "
          f"num={args.num}）===")
    header = (
        f"{'mode':6} {'wall_ms':>10} {'req/s':>9} {'e2e_p50':>9} {'e2e_p95':>9} "
        f"{'e2e_p99':>9} {'q_p50':>8} {'batches':>8} {'avg_bs':>7} {'avg_pad':>8}"
    )
    print(header)
    for run in report["runs"]:
        summary = run["batch_summary"]
        sizes = [
            r for r in summary.get("batch_size", {}).get("distribution", [])
        ] if summary.get("num_batches") else []
        avg_bs = (
            summary["num_requests"] / summary["num_batches"]
            if summary.get("num_batches")
            else float("nan")
        )
        avg_pad = summary.get("padding_ratio", {}).get("avg", float("nan"))
        e2e = run["e2e_ms"]
        print(
            f"{run['mode']:6} {run['wall_ms']:10.1f} {run['requests_per_s']:9.1f} "
            f"{e2e['p50']:9.2f} {e2e['p95']:9.2f} {e2e['p99']:9.2f} "
            f"{run['queue_ms']['p50']:8.2f} "
            f"{summary.get('num_batches', 0):8d} {avg_bs:7.1f} {avg_pad:8.3f}"
        )
    for mode, check in report["correctness"].items():
        verdict = (
            "PASS"
            if check.get("min_cosine", 0) >= 0.999 and check.get("max_abs_diff", 1) <= 0.05
            else "FAIL"
        )
        check["verdict"] = verdict
        print(f"正确性 {mode} vs A: {check}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        f"{args.executor}-{args.scenario}-n{args.num}-s{args.seed}.json"
    )
    slim = json.loads(json.dumps(report, default=str))
    out.write_text(json.dumps(slim, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
