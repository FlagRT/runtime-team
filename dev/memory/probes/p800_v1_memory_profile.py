#!/usr/bin/env python
"""P800 V1 显存画像探针 v2 (memory 子方向 / 重建版)

对应 910c V1 画像 (docs/V1-显存画像报告-20260817.md) 的最小复刻:
  阶段1 加载    : 模型加载耗时 + 加载后 HBM 占用
  阶段2 预热    : 短请求, 避开首次 attention/autotune 慢路径
  阶段3 画像    : 递增输入长度 (128/1k/4k/8k tokens), 每轮采样 HBM
  阶段4 并发    : 4x2048-token 并发, 观测峰值 (910c P0 复现点)

v2 改进: 单阶段超时 (默认 120s, 超时即退出并保留已写 CSV) + 增量写 CSV + 阶段选择
方法学: 与 910c 对齐 (预热必做; 设备级 HBM 采样交叉验证见 benchmarks/xpu_smi_sampler.sh)

用法:
    source /root/miniconda/bin/activate python310_torch29_cuda
    CUDA_VISIBLE_DEVICES=1 python p800_v1_memory_profile.py [模型路径] [阶段: all|load|length|concurrent] [阶段超时秒]
"""
import os
import sys
import time
import csv
import signal

# ---- 必须在 import vllm/flag_gems 之前设置 (默认 flagos 路径; 可用 env 覆盖为 vendor) ----
os.environ.setdefault("VLLM_FL_PLATFORM", "kunlunxin")
os.environ.setdefault("VLLM_FL_PREFER", "flagos")
os.environ.setdefault("USE_FLAGGEMS", "1")
os.environ.setdefault("GEMS_VENDOR", "kunlunxin")
os.environ.setdefault("KLX_USE_AUTOTUNE", "0")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen2.5-1.5B-Instruct"
PHASE = (sys.argv[2] if len(sys.argv) > 2 else "all").lower()
PHASE_CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 120  # 单阶段硬超时(秒)

OUT_DIR = "/workspace/dev/memory/benchmarks/out"
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = f"{OUT_DIR}/v1_profile_p800.csv"

SENT = "显存管理是运行时系统的重要组成部分，它决定了模型推理的峰值占用与吞吐表现。"
TARGET_LENS = [128, 1024, 4096, 8192]


class PhaseTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise PhaseTimeout()


signal.signal(signal.SIGALRM, _timeout_handler)

rows = []


def record(row: dict):
    rows.append(row)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


def build_prompt(tokenizer, n_tokens: int) -> str:
    full = (SENT * (n_tokens // 8 + 4))[: n_tokens * 8]
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def hbm_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


def run_phase(llm, name, prompts, sp, cap=PHASE_CAP):
    signal.alarm(cap)
    t0 = time.time()
    try:
        llm.generate(prompts, sp)
        dt = time.time() - t0
        signal.alarm(0)
        row = {"phase": name, "status": "ok", "elapsed_s": round(dt, 2),
               "hbm_gib": round(hbm_gib(), 2)}
        record(row)
        print(f"[V1] {name}: {dt:.1f}s, HBM {row['hbm_gib']:.2f} GiB", flush=True)
    except PhaseTimeout:
        signal.alarm(0)
        row = {"phase": name, "status": f"TIMEOUT>{cap}s", "elapsed_s": cap,
               "hbm_gib": round(hbm_gib(), 2)}
        record(row)
        print(f"[V1] {name}: 超时 {cap}s (疑似 910c P0 同款卡死), 已保留部分数据, 退出", flush=True)
        sys.exit(2)


def main():
    print(f"[V1] model={MODEL} phase={PHASE} cap={PHASE_CAP}s", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    llm = LLM(
        model=MODEL,
        max_num_batched_tokens=16384,
        max_num_seqs=256,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
    )
    record({"phase": "load", "status": "ok", "elapsed_s": 0,
            "hbm_gib": round(hbm_gib(), 2)})
    print(f"[V1] 加载后 HBM 占用 {hbm_gib():.2f} GiB", flush=True)

    if PHASE in ("all", "warmup"):
        run_phase(llm, "warmup", ["hi"] * 2, SamplingParams(max_tokens=8, temperature=0.0))

    if PHASE in ("all", "length"):
        for n in TARGET_LENS:
            prompt = build_prompt(tokenizer, n)
            run_phase(llm, f"length_{n}", [prompt],
                      SamplingParams(max_tokens=32, temperature=0.0))

    if PHASE in ("all", "concurrent"):
        prompt = build_prompt(tokenizer, 2048)
        sp = SamplingParams(max_tokens=64, temperature=0.0)
        run_phase(llm, "concurrent_4x2048", [prompt] * 4, sp)

    print(f"[V1] 完成, 结果: {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
