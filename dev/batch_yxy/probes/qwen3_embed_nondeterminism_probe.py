#!/usr/bin/env python
"""vllm-ascend 0.20.2rc1 + Qwen3-Embedding-0.6B @ 910C pooling 非确定缺陷复现探针。

发现（2026-09-23，详见 docs/note_动态组批接入说明.md §7）：
- 同一 embed() 调用内结果自洽（同文本 4 连发 cos=1.0）；
- 跨调用相同文本向量漂移（min_cos 低至 -0.04，随并发/规模加重）；
- 返回顺序=输入顺序契约成立（反序调用逐位对得上）；
- enable_chunked_prefill=False 崩溃（select_common_block_size）；
- dtype=float32 崩溃（CANN Nnopbase executor）。

用法：
  .venv/bin/python probes/qwen3_embed_nondeterminism_probe.py \
      --model /mnt/raid/hliu553/models/Qwen3-Embedding-0.6B [--enforce-eager] \
      [--max-num-seqs N] [--num 32] [--out result.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cos(a, b) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def build_texts(tok, num: int, seed: int = 7):
    import numpy as np

    rng = np.random.default_rng(seed)
    words = [f"w{j}" for j in range(1024)]
    texts = []
    for i in range(num):
        n = int(rng.integers(16, 65)) if rng.random() < 0.5 else int(rng.integers(512, 2049))
        start = (i * 37) % 1000
        t = " ".join(words[(start + j) % 1024] for j in range(n))
        while len(tok.encode(t)) > 2048:
            t = " ".join(t.split()[: int(len(t.split()) * 0.9)])
        texts.append(t)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
    ap.add_argument("--num", type=int, default=32)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--enable-prefix-caching", action="store_true")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import numpy as np
    from vllm import LLM

    kwargs = dict(
        model=args.model,
        max_model_len=4096,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    texts = build_texts(tok, args.num)

    def embed(ts):
        outs = llm.embed(list(ts), use_tqdm=False)
        return [np.asarray(o.outputs.embedding, dtype=np.float32) for o in outs]

    result = {"config": {k: str(v) for k, v in kwargs.items()}, "num": args.num}
    r1 = [embed([t])[0] for t in texts]
    r2 = [embed([t])[0] for t in texts]
    singles_bad = [i for i in range(args.num) if cos(r1[i], r2[i]) < 0.9999]
    result["singles_repeat_bad"] = singles_bad

    b1, b2 = embed(texts), embed(texts)
    result["batch_repeat_min_cos"] = min(cos(a, b) for a, b in zip(b1, b2))
    per = [cos(a, s) for a, s in zip(b1, r1)]
    result["batch_vs_singles_min_cos"] = min(per)
    result["batch_vs_singles_bad_lt_0.99"] = int(sum(1 for c in per if c < 0.99))

    rev = embed(texts[::-1])[::-1]
    result["order_contract_min_cos"] = min(cos(a, b) for a, b in zip(b1, rev))

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
