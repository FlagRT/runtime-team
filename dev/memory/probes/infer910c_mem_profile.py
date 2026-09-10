#!/usr/bin/env python3
"""infer910c_mem_profile.py —— 锁定推理镜像（vllm-ascend:v0.20.2rc1-a3）显存画像 harness。

用途
  在 910C 锁定推理镜像内，对 `Qwen/Qwen3-Embedding-0.6B`（EMBEDDING 模型，pooling
  runner，无 decode、几乎无 KV 增长）做一轮显存画像：加载阶段结构 + 运行阶段峰值输入。
  产出一份 JSON（config echo + 各段计时 + vLLM 日志解析字段 + driver 侧 torch_npu 计数）。

方法学（沿用 archive/V1-显存画像报告-20260817）
  - EngineCore 是 spawn 子进程：**driver 进程读到的 torch_npu.npu.memory_* 近似为 0**，
    仅作存在性/上下文佐证。真实 HBM 峰值来自外挂 infer910c_hbm_sampler.py（宿主 npu-smi）
    + 本脚本解析的 vLLM 日志字段（Available KV cache memory / GPU KV cache size /
    Maximum concurrency / model weights / peak memory）。
  - **必须先预热**：短请求先跑 --warmup 轮，避开首次 kernel 初始化长尾，再测量。
  - vLLM 日志经 fd 级重定向落盘后解析（子进程日志也能抓到）；也可用 --vllm-log 指定
    外部已 tee 的日志文件跳过重定向。

模式
  offline      : 进程内 vllm.LLM(runner="pooling")，embed 预热 batch + 测量 batch。
  server-probe : 打已在跑的 OpenAI 兼容 /v1/embeddings 端点（--base-url），只测 client 侧
                 时延/吞吐 + 可选 /metrics；无 driver torch_npu 计数（异进程）。

容器运行命令（UNTESTED —— pending 910C 锁定镜像验证）
  # 宿主侧起容器（带卡容器并发上限 3，起之前 docker ps 数一下；两条腿串行）
  docker run -d --name flagos-proto-infer-910c \
    --network host --shm-size 512g --cap-add SYS_PTRACE \
    <per stack.lock: /dev/davinci* + davinci_manager + devmm_svm + hisi_hdc, driver bind> \
    #   即：for i in $(seq 0 15); do --device /dev/davinci$i; done \
    #       --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
    #       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /mnt/raid:/mnt/raid -v $PWD:/workspace -w /workspace \
    quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 sleep infinity

  # 容器内跑 harness（driver 侧计数近 0 属预期，真值看外挂 sampler）
  docker exec flagos-proto-infer-910c bash -lc '
    cd /workspace && python3 dev/memory/probes/infer910c_mem_profile.py \
      --mode offline --runner pooling \
      --model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B \
      --gpu-mem-util 0.9 --max-num-seqs 256 --max-model-len 8192 \
      --batch 64 --seq-len 512 --warmup 3 \
      --alloc-conf expandable_segments:True \
      --tag gmu0.9-eager \
      --out dev/memory/benchmarks/out/infer910c_mem_gmu0.9.json'

  # 同时在宿主机跑：
  #   python3 dev/memory/probes/infer910c_hbm_sampler.py --chips 0 --interval 0.5 \
  #     --out dev/memory/benchmarks/out/infer910c_hbm.csv --tag gmu0.9-eager

模型权重（宿主已就位，任选其一）：
  /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B                       （snapshot 目录，默认）
  /mnt/raid/kangkai/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B   （HF hub cache 布局）

py_compile 说明：torch_npu / vllm 均**不在模块顶层 import**，全部延迟到函数内，
`python3 -m py_compile` 可在无 NPU 环境通过。

UNTESTED —— pending 910C 锁定镜像验证。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

DEFAULT_MODEL = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B"

# 短预热串（避开首次 kernel 初始化长尾；embedding 不产生 decode）
WARMUP_TEXTS = [
    "hello world",
    "运行时层显存画像预热短请求",
    "the quick brown fox",
]

# ---- vLLM 日志字段解析：正则 → JSON key --------------------------------------
# 兼容 0.20.2 常见措辞；实机若行文有出入，按实际日志补 pattern。
LOG_PATTERNS: dict[str, re.Pattern] = {
    # "Available KV cache memory: 12.34 GiB"
    "available_kv_cache_memory": re.compile(
        r"Available KV cache memory[:=]?\s*([0-9.]+)\s*(GiB|GB|MiB|MB)", re.I),
    # "GPU KV cache size: 123,456 tokens"
    "gpu_kv_cache_size_tokens": re.compile(
        r"GPU KV cache size[:=]?\s*([0-9,]+)\s*tokens", re.I),
    # "Maximum concurrency for 8,192 tokens per request: 12.34x"
    "maximum_concurrency": re.compile(
        r"Maximum concurrency(?:\s+for\s+([0-9,]+)\s*tokens[^:]*)?[:=]?\s*([0-9.]+)x", re.I),
    # "Model weights take 1.14 GiB" / "model weight ... 1.14GiB"
    "model_weights_gib": re.compile(
        r"[Mm]odel weights?(?:\s+take)?[^0-9]{0,20}([0-9.]+)\s*(GiB|GB|MiB|MB)"),
    # "peak memory ... 2.34 GiB" / "Peak memory usage ... 2.34GiB"
    "peak_memory_gib": re.compile(
        r"[Pp]eak(?:\s+\w+){0,3}\s+memory[^0-9]{0,20}([0-9.]+)\s*(GiB|GB|MiB|MB)"),
    # non_torch / activation lines occasionally present
    "non_torch_memory_gib": re.compile(
        r"non[_ -]?torch(?:\s+\w+){0,2}[^0-9]{0,20}([0-9.]+)\s*(GiB|GB|MiB|MB)", re.I),
    # "the current vLLM instance can use total_gpu_memory (64.00GiB) x ..."
    "profiled_total_gpu_memory": re.compile(
        r"total_gpu_memory\s*\(?\s*([0-9.]+)\s*(GiB|GB|MiB|MB)", re.I),
}


def _to_gib(val: str, unit: str) -> float:
    v = float(val.replace(",", ""))
    u = unit.lower()
    if u in ("gib", "gb"):
        return v
    if u in ("mib", "mb"):
        return v / 1024.0
    return v


def parse_vllm_log(path: str) -> dict:
    """扫 vLLM 日志文本，抽显存/KV 相关行。缺字段留 None。"""
    fields: dict = {k: None for k in LOG_PATTERNS}
    fields["_raw_matched_lines"] = []
    try:
        with open(path, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        fields["_error"] = f"读日志失败: {e}"
        return fields
    for line in text.splitlines():
        for key, pat in LOG_PATTERNS.items():
            m = pat.search(line)
            if not m:
                continue
            fields["_raw_matched_lines"].append(line.strip()[:300])
            if key == "gpu_kv_cache_size_tokens":
                fields[key] = int(m.group(1).replace(",", ""))
            elif key == "maximum_concurrency":
                fields[key] = float(m.group(m.lastindex))
                if m.group(1):
                    fields["maximum_concurrency_for_tokens"] = int(m.group(1).replace(",", ""))
            else:
                fields[key] = _to_gib(m.group(1), m.group(2))
    return fields


class _LogCapture:
    """fd 级重定向 stdout/stderr 到文件（子进程日志也能抓）；退出时还原。

    --vllm-log 给定外部日志时不启用（passthrough）。
    """

    def __init__(self, path: str | None):
        self.path = path
        self._saved = None
        self._fh = None

    def __enter__(self):
        if not self.path:
            return self
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self._fh = open(self.path, "w")
        self._saved = (os.dup(1), os.dup(2))
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._fh.fileno(), 1)
        os.dup2(self._fh.fileno(), 2)
        return self

    def __exit__(self, *exc):
        if not self.path:
            return False
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        os.close(self._saved[0])
        os.close(self._saved[1])
        self._fh.close()
        return False


def driver_torch_npu_stats() -> dict:
    """driver 进程侧 torch_npu 显存计数。EngineCore 在子进程 → 这里基本为 0，仅佐证。"""
    out: dict = {"_note": "driver 进程口径；EngineCore 为 spawn 子进程，故通常≈0，真值见外挂 npu-smi sampler + vLLM 日志"}
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
    except Exception as e:  # noqa: BLE001
        out["_error"] = f"torch_npu 不可用: {e}"
        return out
    try:
        dev = 0
        out["memory_allocated"] = int(torch_npu.npu.memory_allocated(dev))
        out["max_memory_allocated"] = int(torch_npu.npu.max_memory_allocated(dev))
        out["memory_reserved"] = int(torch_npu.npu.memory_reserved(dev))
        out["max_memory_reserved"] = int(torch_npu.npu.max_memory_reserved(dev))
    except Exception as e:  # noqa: BLE001
        out["_error"] = f"读取 npu.memory_* 失败: {e}"
    return out


def build_llm(model: str, runner: str, args) -> tuple:
    """构造 vllm.LLM；先试 runner=，TypeError 再回退 task="embed"。返回 (llm, how)。"""
    from vllm import LLM

    common = dict(
        model=model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=True,
    )
    try:
        return LLM(runner=runner, **common), f'runner="{runner}"'
    except TypeError:
        return LLM(task="embed", **common), 'task="embed"'


def make_batch(n: int, seq_len: int) -> list[str]:
    """造 n 条约 seq_len token 的输入（粗略按 ~1 token/词 估算，够画像用）。"""
    unit = "运行时层显存画像 memory runtime hbm profile embedding pooling "
    reps = max(1, seq_len // 8 + 2)
    s = (unit * reps)
    return [s for _ in range(n)]


def run_offline(args, result: dict) -> None:
    from vllm import LLM  # noqa: F401  (确保可 import；实际构造在 build_llm)

    t = result["timings"]

    t0 = time.time()
    llm, how = build_llm(args.model, args.runner, args)
    t["load_s"] = round(time.time() - t0, 3)
    result["llm_construction"] = how

    warm = WARMUP_TEXTS * ((args.warmup // len(WARMUP_TEXTS)) + 1)
    warm = warm[: max(1, args.warmup)]
    t0 = time.time()
    for i in range(max(1, args.warmup)):
        _ = llm.embed(warm) if hasattr(llm, "embed") else llm.encode(warm)
    t["warmup_s"] = round(time.time() - t0, 3)

    batch = make_batch(args.batch, args.seq_len)
    t0 = time.time()
    if hasattr(llm, "embed"):
        outs = llm.embed(batch)
    else:
        outs = llm.encode(batch)
    t["measured_s"] = round(time.time() - t0, 3)
    result["measured"] = {
        "batch": args.batch,
        "seq_len_approx": args.seq_len,
        "n_outputs": len(outs),
        "throughput_req_s": round(args.batch / max(1e-6, t["measured_s"]), 2),
    }


def run_server_probe(args, result: dict) -> None:
    import urllib.request

    t = result["timings"]
    base = args.base_url.rstrip("/")
    result["server_probe"] = {"base_url": base}

    def _embed(texts: list[str]) -> float:
        payload = json.dumps({"model": args.model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{base}/v1/embeddings", data=payload,
            headers={"Content-Type": "application/json"})
        s = time.time()
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            r.read()
        return time.time() - s

    t0 = time.time()
    for _ in range(max(1, args.warmup)):
        _embed(WARMUP_TEXTS)
    t["warmup_s"] = round(time.time() - t0, 3)

    batch = make_batch(args.batch, args.seq_len)
    # 简单串行发 concurrency 组，粗测（真正并发压测交给 performance）
    t0 = time.time()
    groups = max(1, args.concurrency)
    per = max(1, args.batch // groups)
    for _ in range(groups):
        _embed(batch[:per])
    t["measured_s"] = round(time.time() - t0, 3)
    result["measured"] = {
        "batch": args.batch, "groups": groups, "per_group": per,
        "throughput_req_s": round((groups * per) / max(1e-6, t["measured_s"]), 2),
    }

    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=args.timeout) as r:
            metrics = r.read().decode(errors="replace")
        keep = [ln for ln in metrics.splitlines()
                if ("kv_cache" in ln or "gpu_cache" in ln or "memory" in ln)
                and not ln.startswith("#")]
        result["server_probe"]["metrics_excerpt"] = keep[:40]
    except Exception as e:  # noqa: BLE001
        result["server_probe"]["metrics_error"] = str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["offline", "server-probe"], default="offline")
    ap.add_argument("--runner", choices=["pooling"], default="pooling")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9, dest="gpu_mem_util")
    ap.add_argument("--max-num-seqs", type=int, default=256, dest="max_num_seqs")
    ap.add_argument("--max-model-len", type=int, default=8192, dest="max_model_len")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=512, dest="seq_len")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="server-probe 分组数；offline 仅记录到 config")
    ap.add_argument("--warmup", type=int, default=3, help="预热轮数（避开首次 kernel 长尾）")
    ap.add_argument("--enforce-eager", action="store_true", dest="enforce_eager",
                    help="禁用 ACLGraph capture（A/B 轴之一）")
    ap.add_argument("--alloc-conf", default="", dest="alloc_conf",
                    help="写入环境变量 PYTORCH_NPU_ALLOC_CONF，如 expandable_segments:True")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", dest="base_url",
                    help="server-probe 模式的 OpenAI 兼容端点")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--tag", default="", help="run 标签，回写 JSON，便于与 sampler CSV 对齐")
    ap.add_argument("--out", required=True, help="JSON 输出路径")
    ap.add_argument("--vllm-log", default="", dest="vllm_log",
                    help="外部已 tee 的 vLLM 日志路径；给定则不做 fd 重定向，直接解析它")
    args = ap.parse_args()

    # 环境：DO_NOT_TRACK 必设（容器内 usage 上报解析 cpuinfo 报错）
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if args.alloc_conf:
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = args.alloc_conf

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # fd 重定向落盘的 vLLM 日志（offline 模式且未指定外部日志时启用）
    capture_log = None
    if args.mode == "offline" and not args.vllm_log:
        capture_log = out_path.rsplit(".", 1)[0] + ".vllm.log"

    result: dict = {
        "schema": "infer910c_mem_profile/v1",
        "status": "untested-pending-910c-validation",
        "tag": args.tag,
        "config": {
            "model": args.model,
            "mode": args.mode,
            "runner": args.runner,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "batch": args.batch,
            "seq_len_approx": args.seq_len,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "enforce_eager": args.enforce_eager,
            "alloc_conf": args.alloc_conf,
            "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
        },
        "env": {
            "image_expected": "quay.io/ascend/vllm-ascend:v0.20.2rc1-a3",
            "container_name_expected": "flagos-proto-infer-910c",
        },
        "timings": {},
        "vllm_log": None,
        "driver_torch_npu_stats": None,
        "errors": [],
    }

    t_start = time.time()
    try:
        with _LogCapture(capture_log):
            if args.mode == "offline":
                run_offline(args, result)
            else:
                run_server_probe(args, result)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"{type(e).__name__}: {e}")
        result["status"] = "run-error"

    result["timings"]["total_s"] = round(time.time() - t_start, 3)

    # driver 侧 torch_npu 计数（offline 才有意义）
    if args.mode == "offline":
        result["driver_torch_npu_stats"] = driver_torch_npu_stats()

    log_path = args.vllm_log or capture_log
    if log_path and os.path.exists(log_path):
        result["vllm_log"] = parse_vllm_log(log_path)
        result["vllm_log"]["_source"] = log_path

    with open(out_path, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"[done] JSON → {out_path}")
    if capture_log:
        print(f"[done] vLLM 日志 → {capture_log}")
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
