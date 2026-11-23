"""规模判别：缓存关闭下，基准同款规模（64条 mixed 64-2048tok）是否损坏。"""
import numpy as np

MODEL = "/work/models/qwen3-embed-0.6b-xpu"


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL, task="embed", max_model_len=4096,
        dtype="float16", enforce_eager=True,
        enable_prefix_caching=False,
    )

    def E(texts):
        out = llm.embed(list(texts), use_tqdm=False)
        return [np.asarray(o.outputs.embedding) for o in out]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    def make(target, seed_word):
        words = [seed_word] * (target + 8)
        text = " ".join(words)
        while len(tok.encode(text)) > target:
            words = words[:-1]
            text = " ".join(words)
        return text

    # 基准 mixed 场景同款长度分布（bounds 64..2048）
    lengths = [64, 96, 128, 160, 200, 256, 300, 384, 448, 512,
               640, 768, 900, 1024, 1200, 1536, 1800, 2048] * 2  # 36条
    lengths += [64] * 14 + [2048] * 14  # 64条，两端加密
    texts = [make(L, f"w{i}") for i, L in enumerate(lengths)]

    refs = [E([t])[0] for t in texts]           # 单条参照（缓存关，无跨调用干扰）

    # 一次 64 条混批（= 基准 A 路径）
    V = E(texts)
    coss = [cos(v, r) for v, r in zip(V, refs)]
    bad = [(lengths[i], round(c, 4)) for i, c in enumerate(coss) if c < 0.999]
    print(f"[64条混批 vs 单条] 最小cos={min(coss):.4f} 损坏条数={len(bad)}/{len(texts)}")
    if bad:
        print(f"  损坏样本(长度,cos): {bad[:10]}")

    # 分桶批（= 基准 C 路径形态：同长度接近的放一批）
    by_len = {}
    for i, L in enumerate(lengths):
        by_len.setdefault(L, []).append(i)
    bad_c = 0
    for L, idxs in sorted(by_len.items()):
        vecs = E([texts[i] for i in idxs])
        for i, v in zip(idxs, vecs):
            if cos(v, refs[i]) < 0.999:
                bad_c += 1
    print(f"[同长度分桶批 vs 单条] 损坏条数={bad_c}/{len(texts)}")


if __name__ == "__main__":
    main()
