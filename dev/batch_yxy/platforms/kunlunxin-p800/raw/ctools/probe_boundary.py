"""安全边界定位：损坏与 (单步请求条数 × prefill步数) 的关系。

矩阵: N∈{8,16,32,64} × 64tok（单步内）+ N∈{8,16,32} × 2048tok（跨1~4步）
缓存关闭，全部对照各自单条参照。
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
        enable_prefix_caching=False,  # 默认预算 max_num_batched_tokens=16384
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

    def trial(n, length):
        texts = [make(length, f"t{n}i{i}") for i in range(n)]
        refs = [E([t])[0] for t in texts]
        V = E(texts)
        bad = sum(1 for v, r in zip(V, refs) if cos(v, r) < 0.999)
        total = n * length
        steps = max(1, -(-total // 16384))
        print(f"  {n:>2}条×{length:>4}tok (总{total:>6}tok≈{steps}步): 损坏 {bad}/{n}")

    print("\n[边界矩阵]（缓存关，prefill预算16384）")
    for n in (8, 16, 32, 64):
        trial(n, 64)
    for n in (8, 16, 32):
        trial(n, 2048)


if __name__ == "__main__":
    main()
