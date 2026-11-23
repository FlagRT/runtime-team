"""Part B（跨步×生成）独立进程版：缓存关，16×2048tok 逐条单独调用 vs 一次批量调用（2步）。"""

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
              dtype="float16", enforce_eager=True,
              enable_prefix_caching=False)
    seeds = ["乙", "丙", "丁", "戊", "己", "庚", "辛", "壬",
             "癸", "子", "丑", "寅", "卯", "辰", "巳", "午"]
    prompts = [make(2048, s) for s in seeds]
    print(f"\n[Part B] 16×2048tok（缓存关）：逐条单独调用 vs 一次批量调用（2步）")
    solo = [llm.generate([p], sp)[0].outputs[0].token_ids for p in prompts]
    batch_out = llm.generate(prompts, sp)
    bad = []
    for i, o in enumerate(batch_out):
        if o.outputs[0].token_ids != solo[i]:
            bad.append(i)
    print(f"  输出与单独调用不一致的提示: {bad}（{len(bad)}/16）"
          f"{' ← 步2(第8-15条)全中招' if bad == list(range(8, 16)) else ''}")


if __name__ == "__main__":
    main()
