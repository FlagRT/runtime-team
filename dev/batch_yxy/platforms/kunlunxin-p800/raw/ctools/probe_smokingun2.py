"""铁证实验v2：在能触发的会话状态下抓损坏项，然后交叉比对。

策略：单条参照 → 同一批 16×2048 连发 4 次（每次改变分配器状态）
      → 32×2048 变体（4步） → 对所有损坏项做全批交叉比对。
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

    all_bad = []  # (round_tag, i, V_i)
    for rnd in range(1, 5):
        V = E(texts)
        own = [cos(v, r) for v, r in zip(V, refs)]
        bad = [i for i, c in enumerate(own) if c < 0.999]
        print(f"[第{rnd}轮 16×2048] 损坏 {len(bad)}/16 {bad}")
        for i in bad:
            all_bad.append((f"16x第{rnd}轮", i, V[i]))

    # 32×2048 变体（复用前16条参照 + 新增16条）
    seeds2 = seeds + ["午", "未", "申", "酉", "戌", "亥", "元", "亨",
                      "利", "贞", "天", "地", "玄", "黄", "宇", "宙"]
    texts32 = [make(2048, s) for s in seeds2]
    refs32 = [E([t])[0] for t in texts32]
    V = E(texts32)
    own = [cos(v, r) for v, r in zip(V, refs32)]
    bad = [i for i, c in enumerate(own) if c < 0.999]
    print(f"[32×2048] 损坏 {len(bad)}/32 {bad}")
    for i in bad:
        all_bad.append(("32x", i, V[i]))

    print(f"\n[交叉比对] 共 {len(all_bad)} 个损坏样本")
    for tag, i, v in all_bad[:12]:
        coss = sorted(((j, cos(v, refs32[j])) for j in range(len(refs32))),
                      key=lambda x: -x[1])
        top = ", ".join(f"ref{j}={c:+.4f}" for j, c in coss[:3])
        rank = [j for j, _ in coss].index(i) + 1
        print(f"  {tag} 损坏项{i}: Top3 {top} | 自身参照排第{rank}")


if __name__ == "__main__":
    main()
