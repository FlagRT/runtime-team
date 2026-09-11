#!/usr/bin/env python3
"""infer910c_ab_matrix.py —— infer910c_mem_profile.py 的 A/B 矩阵驱动 + 结果汇总。

用途
  为「显存池对照数据」交付物（战略文档 §5 memory 第 2 条）批量跑 infer910c_mem_profile.py，
  每个 A/B 点起一次子进程，各自出一份 JSON，最后收敛成一张 CSV + 一张 Markdown 表。

A/B 轴（--axis 选一）
  gpu-mem-util   : 0.4 vs 0.9
      vLLM 一次性预分配比例。embedding 模型 KV/pooling 池可能接近 0，这条主要看
      「预分配上限 - 实占」的余量差、以及加载后 HBM used 是否随该比例线性变化，
      给 device-context / 调度 一个安全 gpu_mem_util 区间。
  alloc-conf     : expandable_segments:True vs 默认
      PYTORCH_NPU_ALLOC_CONF。expandable_segments 改变 caching allocator 段增长策略，
      直接影响 reserved-vs-allocated 碎片量 —— 显存池「底座」层的关键对照。
  enforce-eager  : True vs ACLGraph（默认）
      ACLGraph capture 会占一块图工作区（旧栈 graph 模式实测多占 ~7GiB 级 KV 空间）。
      eager 换「零成本扩容」但吞吐下降，量化二者显存/时延互换。
  max-num-seqs   : sweep（默认 32,64,128,256）
      批上限影响激活区峰值与调度器预留。给调度侧一个安全 max_num_seqs 上界。

用法（UNTESTED —— pending 910C 锁定镜像验证；须在锁定推理容器 flagos-proto-infer-910c 内跑）
  docker exec flagos-proto-infer-910c bash -lc '
    cd /workspace && python3 dev/memory/probes/910c/mem-ab-matrix_910c.py \
      --axis gpu-mem-util \
      --model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
      --batch 64 --seq-len 512 --warmup 3 \
      --outdir dev/memory/benchmarks/out/ab_gmu'
  # 建议：宿主机同时跑 infer910c_hbm_sampler.py，--tag 用本脚本打印的 run tag 对齐 HBM 峰值。

产物
  <outdir>/run_<axis>_<value>.json     每个 A/B 点的 profile JSON
  <outdir>/run_<axis>_<value>.vllm.log vLLM 日志
  <outdir>/ab_summary.csv              汇总表
  <outdir>/ab_summary.md               汇总表（Markdown，可直接贴报告）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILE = os.path.join(HERE, "infer910c_mem_profile.py")

AXES: dict[str, list[dict]] = {
    "gpu-mem-util": [
        {"label": "gmu0.4", "gpu_mem_util": 0.4},
        {"label": "gmu0.9", "gpu_mem_util": 0.9},
    ],
    "alloc-conf": [
        {"label": "alloc-default", "alloc_conf": ""},
        {"label": "alloc-expandable", "alloc_conf": "expandable_segments:True"},
    ],
    "enforce-eager": [
        {"label": "aclgraph", "enforce_eager": False},
        {"label": "eager", "enforce_eager": True},
    ],
    "max-num-seqs": [
        {"label": "mns32", "max_num_seqs": 32},
        {"label": "mns64", "max_num_seqs": 64},
        {"label": "mns128", "max_num_seqs": 128},
        {"label": "mns256", "max_num_seqs": 256},
    ],
}

# 汇总表列（从各 JSON 抽）
SUMMARY_COLS = [
    "run", "tag", "status",
    "load_s", "warmup_s", "measured_s", "total_s", "throughput_req_s",
    "model_weights_gib", "available_kv_cache_memory",
    "gpu_kv_cache_size_tokens", "maximum_concurrency", "peak_memory_gib",
    "non_torch_memory_gib", "profiled_total_gpu_memory",
    "driver_max_memory_reserved",
]


def build_cmd(args, point: dict, out_json: str) -> list[str]:
    cfg = dict(
        gpu_mem_util=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        alloc_conf=args.alloc_conf,
        enforce_eager=args.enforce_eager,
    )
    cfg.update({k: v for k, v in point.items() if k != "label"})

    cmd = [
        sys.executable, PROFILE,
        "--mode", "offline", "--runner", "pooling",
        "--model", args.model,
        "--gpu-mem-util", str(cfg["gpu_mem_util"]),
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--max-model-len", str(cfg["max_model_len"]),
        "--batch", str(args.batch),
        "--seq-len", str(args.seq_len),
        "--warmup", str(args.warmup),
        "--tag", point["label"],
        "--out", out_json,
    ]
    if cfg.get("alloc_conf"):
        cmd += ["--alloc-conf", cfg["alloc_conf"]]
    if cfg.get("enforce_eager"):
        cmd += ["--enforce-eager"]
    return cmd


def flatten(run_label: str, j: dict) -> dict:
    t = j.get("timings", {}) or {}
    m = j.get("measured", {}) or {}
    log = j.get("vllm_log", {}) or {}
    drv = j.get("driver_torch_npu_stats", {}) or {}
    row = {c: "" for c in SUMMARY_COLS}
    row["run"] = run_label
    row["tag"] = j.get("tag", "")
    row["status"] = j.get("status", "")
    row["load_s"] = t.get("load_s", "")
    row["warmup_s"] = t.get("warmup_s", "")
    row["measured_s"] = t.get("measured_s", "")
    row["total_s"] = t.get("total_s", "")
    row["throughput_req_s"] = m.get("throughput_req_s", "")
    for k in ("model_weights_gib", "available_kv_cache_memory",
              "gpu_kv_cache_size_tokens", "maximum_concurrency",
              "peak_memory_gib", "non_torch_memory_gib", "profiled_total_gpu_memory"):
        row[k] = log.get(k, "")
    row["driver_max_memory_reserved"] = drv.get("max_memory_reserved", "")
    return row


def write_md(rows: list[dict], path: str, axis: str) -> None:
    with open(path, "w") as f:
        f.write(f"# infer910c A/B 汇总 —— 轴: `{axis}`\n\n")
        f.write("> UNTESTED —— pending 910C 锁定镜像验证。HBM 峰值真值请对齐 infer910c_hbm_sampler.py CSV（按 tag）。\n\n")
        f.write("| " + " | ".join(SUMMARY_COLS) + " |\n")
        f.write("| " + " | ".join("---" for _ in SUMMARY_COLS) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in SUMMARY_COLS) + " |\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", required=True, choices=list(AXES))
    ap.add_argument("--model", default="/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
    ap.add_argument("--outdir", required=True)
    # 非目标轴的固定值
    ap.add_argument("--gpu-mem-util", type=float, default=0.9, dest="gpu_mem_util")
    ap.add_argument("--max-num-seqs", type=int, default=256, dest="max_num_seqs")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    ap.add_argument("--alloc-conf", default="", dest="alloc_conf")
    ap.add_argument("--enforce-eager", action="store_true", dest="enforce_eager")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512, dest="seq_len")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="只打印将执行的命令，不跑")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    points = AXES[args.axis]
    rows: list[dict] = []

    for point in points:
        run_label = f"{args.axis}_{point['label']}"
        out_json = os.path.join(args.outdir, f"run_{run_label}.json")
        cmd = build_cmd(args, point, out_json)
        print(f"\n=== {run_label} ===\n$ {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.run(cmd).returncode
        print(f"[{run_label}] exit={rc}")
        if os.path.exists(out_json):
            with open(out_json) as f:
                rows.append(flatten(run_label, json.load(f)))
        else:
            rows.append({**{c: "" for c in SUMMARY_COLS}, "run": run_label,
                         "status": f"no-json(exit={rc})"})

    if args.dry_run:
        return 0

    csv_path = os.path.join(args.outdir, "ab_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)
    md_path = os.path.join(args.outdir, "ab_summary.md")
    write_md(rows, md_path, args.axis)
    print(f"\n[done] {csv_path}\n[done] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
