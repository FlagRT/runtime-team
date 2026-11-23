MODEL = "/root/qwen3-gen"


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)

    def make(target, seed_word):
        words = [seed_word] * (target + 8)
        text = " ".join(words)
        while len(tok.encode(text)) > target:
            words = words[:-1]
            text = " ".join(words)
        return text

    sp = SamplingParams(temperature=0.0, max_tokens=32)
    llm = LLM(model=MODEL, task="generate", max_model_len=4096,
              dtype="float16", enforce_eager=True, enable_prefix_caching=False)
    seeds = ["乙", "丙", "丁", "戊", "己", "庚", "辛", "壬",
             "癸", "子", "丑", "寅", "卯", "辰", "巳", "午"]
    prompts = [make(2048, s) for s in seeds]
    solo = [llm.generate([p], sp)[0].outputs[0].token_ids for p in prompts]
    batch = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]
    print("[分叉量级] 每条: 首个分叉位置 / 32tok 中相同数")
    for i in range(16):
        a, b = solo[i], batch[i]
        same = sum(1 for x, y in zip(a, b) if x == y)
        first = next((k for k in range(len(a)) if a[k] != b[k]), len(a))
        print(f"  条{i:>2}: 首分叉@{first:>2}  相同{same}/32")


if __name__ == "__main__":
    main()
