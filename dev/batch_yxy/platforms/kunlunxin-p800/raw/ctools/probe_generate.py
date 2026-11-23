"""生成路径验证：缓存命中与跨步缺陷是否影响 task=generate。

Part A（缺陷一·生成版）: 缓存开，同一 ~400tok 长提示贪心解码两次
     第一次冷缓存 / 第二次全量命中 → 若 token 序列分叉 = 生成路径中招
Part B（缺陷二·生成版）: 缓存关，16×2048tok 提示一次调用（2步） vs 逐条单独调用（各1步）
     若第二批（步2）的输出与单独调用不同 = 生成路径跨步也中招
"""

import numpy as np

MODEL = "/work/models/qwen3-embed-0.6b-xpu"


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

    # ---- Part A: 缓存命中 × 生成 ----
    llm = LLM(model=MODEL, task="generate", max_model_len=4096,
              dtype="float16", enforce_eager=True)  # 缓存默认开
    prompt = make(400, "甲")
    print(f"\n[Part A] 提示 {len(tok.encode(prompt))} tok，贪心解码两次（缓存开）")
    o1 = llm.generate([prompt], sp)[0].outputs[0]
    o2 = llm.generate([prompt], sp)[0].outputs[0]  # 第二次全量命中
    same = o1.token_ids == o2.token_ids
    diff_at = next((k for k, (a, b) in enumerate(zip(o1.token_ids, o2.token_ids)) if a != b), None)
    print(f"  第一次: {o1.token_ids[:10]}...")
    print(f"  第二次: {o2.token_ids[:10]}...")
    print(f"  token序列相同: {same}" + ("" if same else f"（从第{diff_at}个token分叉）"))
    print(f"  文本1: {o1.text[:60]!r}")
    print(f"  文本2: {o2.text[:60]!r}")
    del llm

    # ---- Part B: 跨步 × 生成 ----
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
          f"{'← 步2(第8-15条)全中招' if bad == list(range(8, 16)) else ''}")


if __name__ == "__main__":
    main()
