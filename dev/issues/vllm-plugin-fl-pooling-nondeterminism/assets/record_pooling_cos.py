#!/usr/bin/env python3
"""现象记录脚本（收数），不是判定脚本：只记录两次独立 embed() 的 cos。

0=收数完成，2=执行出错；均不代表修复验收结论。--disable-prefix-caching 为对照组。
运行前须 shell export 三个网络变量；需先解决 pooling dispatch 断言。
自动建立剔除 sentence-transformers 两个配置的临时模型视图，不改原目录。
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/models/Qwen3-Embedding-0.6B/")
    parser.add_argument("--disable-prefix-caching", action="store_true")
    args = parser.parse_args()
    try:
        for key, value in {"GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo",
                           "VLLM_HOST_IP": "127.0.0.1"}.items():
            if os.environ.get(key) != value:
                raise RuntimeError(f"shell export required: {key}={value}")
        from vllm import LLM
        import numpy as np

        source = Path(args.model).resolve(strict=True)
        with tempfile.TemporaryDirectory(prefix="pooling-cos-model-") as view:
            for entry in source.iterdir():
                if entry.name not in {"config_sentence_transformers.json", "modules.json"}:
                    Path(view, entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
            options = {"enable_prefix_caching": False} if args.disable_prefix_caching else {}
            llm = LLM(model=view, runner="pooling", convert="embed", max_model_len=4096,
                      dtype="float16", enforce_eager=True, **options)

            def E(text):
                return np.asarray(llm.embed([text], use_tqdm=False)[0].outputs.embedding)

            v1 = E("你好，世界")
            v2 = E("你好，世界")
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            print(f"COS = {cos:+.6f}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
