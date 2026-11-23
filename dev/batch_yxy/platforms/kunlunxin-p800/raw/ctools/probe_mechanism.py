"""机制验证：256 边界 + uint8 位置回绕假设 + CPU 真值对照。

E5 精确边界: 短10 + 长L, L∈{250..300}，找损坏临界点
E7 CPU真值: transformers 手工 last-token pooling 对照，判定 vllm 哪一侧错
E8 回绕假设: 若内核读了位置 (L mod 256)，则混批中长文本结果 ≈ 其前 (L mod 256) token 前缀的嵌入
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
        return text, len(tok.encode(text))

    # ---- E5: 精确边界 ----
    print("\n[E5] 边界扫描：短10 + 长L（cos_short / cos_long 相对各自单条参照）")
    s_text, _ = make(10, "戊")
    ref_s = E([s_text])[0]
    for L in (250, 254, 255, 256, 257, 258, 260, 264, 280, 300):
        lt, actual = make(L, "己")
        ref_l = E([lt])[0]
        pair = E([s_text, lt])
        print(f"  L={actual:>3}: cos_short={cos(pair[0], ref_s):+.4f}  cos_long={cos(pair[1], ref_l):+.4f}")

    # ---- E8: 回绕假设（可证伪预测）----
    print("\n[E8] uint8 回绕假设：混批中300tok结果 vs 其前(300%256=44)tok前缀的嵌入")
    t300, n300 = make(300, "庚")
    # 构造恰好 44 token 的前缀（同一词表词，保证前缀关系）
    prefix_words = t300.split()
    prefix = " ".join(prefix_words)
    while len(tok.encode(prefix)) > 44:
        prefix_words = prefix_words[:-1]
        prefix = " ".join(prefix_words)
    n_prefix = len(tok.encode(prefix))
    ref_prefix = E([prefix])[0]          # 44-token 前缀的单条嵌入
    solo300 = E([t300])[0]
    mixed = E([make(10, "甲")[0], t300])  # 混批：短 + 300
    print(f"  前缀实际token数={n_prefix}（目标44，300%256={300 % 256}）")
    print(f"  cos(混批[300], 前缀嵌入)={cos(mixed[1], ref_prefix):+.4f}   ← 若≈1.0 则回绕假设成立")
    print(f"  cos(单条[300], 前缀嵌入)={cos(solo300, ref_prefix):+.4f}   ← 对照")

    # ---- E7: CPU 真值 ----
    print("\n[E7] CPU transformers 真值（last-token pooling + L2归一）")
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    def truth(text):
        ids = tok(text, return_tensors="pt")
        with torch.no_grad():
            h = model(**ids).last_hidden_state  # 无padding，最后一位即最后token
        v = h[0, -1].numpy().astype(np.float64)
        return v / np.linalg.norm(v)

    t_true = truth(t300)
    print(f"  cos(真值300, vllm单条300)={cos(t_true, solo300):+.4f}")
    print(f"  cos(真值300, vllm混批300)={cos(t_true, mixed[1]):+.4f}")
    print("  → 哪个更接近1.0，哪个就是对的")


if __name__ == "__main__":
    main()
