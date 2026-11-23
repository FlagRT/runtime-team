"""铁证实验v3：用两个已验证的可靠触发配方抓损坏，然后全批交叉比对。

配方A（probe_scale，新鲜会话稳定复现 35/64）: 64条混长 64..2048
配方B（probe_boundary 序列）: 8×64→16×64→32×64→64×64→8×2048→16×2048
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

    def cross_match(tag, texts, refs, V):
        own = [cos(v, r) for v, r in zip(V, refs)]
        bad = [i for i, c in enumerate(own) if c < 0.999]
        print(f"[{tag}] 损坏 {len(bad)}/{len(texts)} {bad[:12]}")
        hits = 0
        for i in bad[:12]:
            coss = sorted(((j, cos(V[i], refs[j])) for j in range(len(refs))),
                          key=lambda x: -x[1])
            top = ", ".join(f"ref{j}={c:+.4f}" for j, c in coss[:3])
            rank = [j for j, _ in coss].index(i) + 1
            exact = any(c > 0.999 for j, c in coss if j != i)
            hits += int(exact)
            print(f"  损坏项{i}: Top3 {top} | 自身排第{rank} | 存在非对角≈1: {exact}")
        return len(bad), hits

    # ---- 配方A: 64条混长（scale 同款）----
    lengths = [64, 96, 128, 160, 200, 256, 300, 384, 448, 512,
               640, 768, 900, 1024, 1200, 1536, 1800, 2048] * 2
    lengths += [64] * 14 + [2048] * 14
    textsA = [make(L, f"w{i}") for i, L in enumerate(lengths)]
    refsA = [E([t])[0] for t in textsA]
    VA = E(textsA)
    cross_match("A: 64条混长", textsA, refsA, VA)

    # ---- 配方B: boundary 序列 ----
    for n, length in ((8, 64), (16, 64), (32, 64), (64, 64), (8, 2048), (16, 2048)):
        texts = [make(length, f"t{n}i{i}") for i in range(n)]
        refs = [E([t])[0] for t in texts]
        V = E(texts)
        cross_match(f"B: {n}×{length}", texts, refs, V)


if __name__ == "__main__":
    main()
