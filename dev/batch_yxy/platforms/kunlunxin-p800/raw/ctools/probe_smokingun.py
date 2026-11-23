"""铁证实验：跨步损坏向量是否 = 批内其他文本的正确嵌入（张冠李戴现行）。

设计：缓存关，16条×2048tok（总32768 > 16384 → 2步）。
每条内容互不相同（不同种子词）。先取单条参照（正确），再一次混批调用。
对每个损坏项，与批内全部 16 条参照交叉比对，找非对角 ≈1.0 的匹配。
"""

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

    seeds = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛",
             "壬", "癸", "子", "丑", "寅", "卯", "辰", "巳"]
    texts = [make(2048, s) for s in seeds]
    refs = [E([t])[0] for t in texts]

    V = E(texts)  # 一次调用 16×2048 → 2 个 prefill 步
    own = [cos(v, r) for v, r in zip(V, refs)]
    bad_idx = [i for i, c in enumerate(own) if c < 0.999]
    print(f"[结果] 16条×2048 一次调用: 损坏 {len(bad_idx)}/16（损坏项: {bad_idx}）")
    print(f"       各项自身cos: {[round(c, 3) for c in own]}")

    print("\n[交叉比对] 每个损坏项 vs 批内全部参照（找非对角≈1.0）")
    for i in bad_idx:
        coss = [(j, cos(V[i], refs[j])) for j in range(len(refs))]
        coss.sort(key=lambda x: -x[1])
        top = ", ".join(f"ref{j}({s})={c:+.4f}" for j, c in coss[:3])
        own_rank = [j for j, _ in coss].index(i)
        print(f"  损坏项{i}(种子{seeds[i]}): Top3 {top} | 自身参照排第{own_rank + 1}")


if __name__ == "__main__":
    main()
