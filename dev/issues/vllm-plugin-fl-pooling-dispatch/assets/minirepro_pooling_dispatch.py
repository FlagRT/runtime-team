#!/usr/bin/env python3
"""Pooling dispatch 复现 / 验收；0=PASS，1=FAIL（目标断言），2=ERROR。

运行前须由 shell export GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
VLLM_HOST_IP=127.0.0.1。父进程收集 worker 及其子进程输出，避免 EngineDeadError
掩盖实际断言。模型自动建立剔除两个 sentence-transformers 配置的临时软链接视图，
不修改原模型目录。不下载模型，不修改插件。
"""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback

DEFAULT_MODEL = "/home/models/Qwen3-Embedding-0.6B/"
NETWORK_ENV = {"GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo",
               "VLLM_HOST_IP": "127.0.0.1"}
EXCLUDED = {"config_sentence_transformers.json", "modules.json"}


def verdict(state, detail, code):
    print("\n=== VERDICT (minirepro_pooling_dispatch, 1 card) ===", flush=True)
    print(f"  repro: {detail}", flush=True)
    print(f"overall: {state}", flush=True)
    return code


def worker(model):
    from vllm import LLM

    source = Path(model).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="pooling-dispatch-model-") as view:
        for entry in source.iterdir():
            if entry.name not in EXCLUDED:
                Path(view, entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        llm = LLM(model=view, runner="pooling", convert="embed", max_model_len=4096,
                  dtype="float16", enforce_eager=True)
        out = llm.embed(["你好，世界"], use_tqdm=False)
        dim = len(out[0].outputs.embedding)
        print("EMBED_OK, dim=", dim, flush=True)
        if dim != 1024:
            raise ValueError(f"unexpected embedding dimension: {dim}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.model)
        return 0
    try:
        for key, value in NETWORK_ENV.items():
            if os.environ.get(key) != value:
                raise RuntimeError(f"shell export required: {key}={value}")
        run = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
             "--model", args.model], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", check=False)
        print(run.stdout, end="", flush=True)
        # 仅匹配 execute_model 的目标栈帧；其他 AssertionError / EngineDeadError 不算复现。
        target = re.search(
            r'File "/workspace/vllm-plugin-FL/vllm_fl/worker/worker\.py", line \d+, in execute_model\s*\n'
            r'[^\n]*assert isinstance\(output, IntermediateTensors\)\s*\n'
            r'[^\n]*AssertionError', run.stdout)
        if run.returncode != 0 and target:
            return verdict("FAIL", "CONFIRMED - execute_model rejected pooling output", 1)
        if run.returncode == 0 and "EMBED_OK, dim= 1024" in run.stdout and "AssertionError" not in run.stdout:
            return verdict("PASS", "NOT_REPRODUCED - EMBED_OK, dim= 1024", 0)
        return verdict("ERROR", "UNDETERMINED - other failure or unexpected output", 2)
    except Exception:
        traceback.print_exc()
        return verdict("ERROR", "UNDETERMINED - environment or launcher failure", 2)


if __name__ == "__main__":
    sys.exit(main())
