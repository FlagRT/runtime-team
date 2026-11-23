"""终验：CPU 真值 + 缓存命中最小复现。"""
import numpy as np

MODEL = "/work/models/qwen3-embed-0.6b-xpu"


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(MODEL)

    # 构造 300 tok 文本（与 probe_prefix 同构）
    words = ["丙"] * 308
    text = " ".join(words)
    while len(tok.encode(text)) > 300:
        words = words[:-1]
        text = " ".join(words)

    # ---- CPU 真值 ----
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        h = model.model(**ids).last_hidden_state  # base model，无padding
    truth = h[0, -1].numpy().astype(np.float64)
    truth = truth / np.linalg.norm(truth)

    # ---- XPU：缓存默认开，同文本连发两次 ----
    llm = LLM(model=MODEL, task="embed", max_model_len=4096,
              dtype="float16", enforce_eager=True)  # 不传 enable_prefix_caching → 默认开

    def E(t):
        return np.asarray(llm.embed([t], use_tqdm=False)[0].outputs.embedding)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    v1 = E(text)  # 第一次：冷缓存，真实 prefill
    v2 = E(text)  # 第二次：全量缓存命中
    print(f"[最小复现] 同一300tok文本连发两次: cos(v1,v2)={cos(v1, v2):+.4f}")
    print(f"[真值对照] cos(CPU真值, 第一次)={cos(truth, v1):+.4f}")
    print(f"[真值对照] cos(CPU真值, 第二次)={cos(truth, v2):+.4f}")


if __name__ == "__main__":
    main()
