"""隔离变量：prefix caching 开/关 × 批组成矩阵。

用法: python probe_prefix.py --caching 0|1
同一组文本（s9a,s9b,l300a,l300b），先取单条参照，再按多种批组成混批，
报告每个位置相对参照的 cos。跨进程对比 caching=0/1 的输出。
"""

import argparse

import numpy as np

MODEL = "/work/models/qwen3-embed-0.6b-xpu"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--caching", type=int, required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL, task="embed", max_model_len=4096,
        dtype="float16", enforce_eager=True,
        enable_prefix_caching=bool(args.caching),
    )

    def E(texts):
        out = llm.embed(list(texts), use_tqdm=False)
        return [np.asarray(o.outputs.embedding) for o in out]

    def cos(a, b):
        d = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / d) if d > 0 else 1.0

    def make(target, seed_word):
        words = [seed_word] * (target + 8)
        text = " ".join(words)
        while len(tok.encode(text)) > target:
            words = words[:-1]
            text = " ".join(words)
        return text

    s9a, s9b = make(9, "甲"), make(9, "乙")
    l300a, l300b = make(300, "丙"), make(300, "丁")
    texts = {"9a": s9a, "9b": s9b, "300a": l300a, "300b": l300b}
    refs = {k: E([t])[0] for k, t in texts.items()}

    def show(tag, keys, vecs):
        parts = "  ".join(f"{k}:{cos(v, refs[k]):+.4f}" for k, v in zip(keys, vecs))
        print(f"  {tag:<28} {parts}")

    print(f"\n[prefix_caching={args.caching}] 批组成矩阵（+1.0000=无损）")
    for tag, keys in [
        ("b1 [9a,300a]", ["9a", "300a"]),
        ("b2 [9a,9b,300a]", ["9a", "9b", "300a"]),
        ("b3 [9a,300a,300b]", ["9a", "300a", "300b"]),
        ("b4 [9a,9b,300a,300b]", ["9a", "9b", "300a", "300b"]),
        ("b5 b4重复(缓存已热)", ["9a", "9b", "300a", "300b"]),
        ("b6 [300a,300b,9a,9b]", ["300a", "300b", "9a", "9b"]),
    ]:
        vecs = E([texts[k] for k in keys])
        show(tag, keys, vecs)


if __name__ == "__main__":
    main()
