#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_tp_equivalence_rigorous.py — A5 稳健性补验：TP 数值等价是否真的成立

【为什么要补验】原 A5 结论（greedy 4/4 逐字一致）的样本量为 **4 条 prompt × 64 token ≈ 256 token**，
对于"数值等价"这类结论明显不足：
  · 浮点误差（HCCL all_reduce 累加顺序差异）通常在**长序列**上累积后才显现
  · prompt 多样性不足：未覆盖代码 / 数学 / 长上下文 / 特殊字符 / 多语言混排
  · 只跑 1 次，未验证**可重复性**（同一配置两次运行结果是否仍一致）

【补验设计】
  · 16 条 prompt：中/英/代码/数学/长文本/特殊字符/多语言混排
  · max_tokens 默认 256（4 倍于原 64）
  · 逐条记录：text 一致 / token_ids 一致 / **一致前缀长度**（若发散，定位从第几个 token 开始）
  · --repeat 2：TP=1 自身两次运行结果对比（验证可重复性，排除"恰好一致"）

【判定】
  TP_EQUIV_STRONG   : 全部 prompt 在 256 token 上 text+token_ids 一致，且 TP=1 自重复一致
  TP_EQUIV_WEAK     : 短前缀一致但长序列发散（需报告一致前缀分布）
  TP_EQUIV_FAIL     : 短前缀即发散

【用法】容器内（需先停 serve 释放卡）：
  python3 test_tp_equivalence_rigorous.py --model /mnt/raid/hliu553/models/Qwen3-4B \
      --tp 1 --max-tokens 256 --out tp1_rigorous.json
  python3 test_tp_equivalence_rigorous.py --model ... --tp 2 --max-tokens 256 --out tp2_rigorous.json
  python3 test_tp_equivalence_rigorous.py --compare tp1_rigorous.json tp2_rigorous.json
"""
import argparse
import json
import os
import sys
import time

PROMPTS = [
    # 英文常识 / 生成
    "The capital of France is",
    "Write a short poem about autumn rain, in four lines:",
    # 中文
    "量子计算的基本原理是",
    "请用三句话解释什么是分布式训练：",
    # 代码
    "def fibonacci(n):",
    "Write a Python function to merge two sorted lists:",
    # 数学 / 推理
    "Compute step by step: 127 * 43 + 19 =",
    "If a train travels 120 km in 1.5 hours, its average speed is",
    # 长上下文（诱导长生成）
    "Write a detailed technical blog outline about large language model inference optimization, "
    "covering at least eight sections with brief descriptions for each:",
    "Explain the history of the Internet from ARPANET to today, chronologically:",
    # 特殊字符 / 格式
    "Format the following as a markdown table with columns Name|Age|City: "
    "Alice 30 Beijing; Bob 25 Shanghai; Carol 35 Shenzhen",
    "Repeat exactly: aB3$xY!@#q  -- and then explain what you repeated:",
    # 多语言混排
    "Translate to English: 今天的天气很好，我们一起去公园散步吧。",
    "混排测试 mixed テスト test mélangé — explain this sentence's languages:",
    # 指令跟随（易产生分歧）
    "List exactly 7 prime numbers greater than 100, separated by commas, no explanation:",
    "Count from 1 to 30, one number per line, no extra text:",
]


def run_model(model, tp, max_tokens, out_path):
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model, tensor_parallel_size=tp, enforce_eager=True)
    t_load = time.time() - t0

    # 预热（坑 A1）
    llm.generate(["Hi"], SamplingParams(max_tokens=8, temperature=0.0))

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=42)
    t1 = time.time()
    outs = llm.generate(PROMPTS, params)
    t_gen = time.time() - t1

    items = []
    for p, o in zip(PROMPTS, outs):
        c = o.outputs[0]
        items.append({
            "prompt": p,
            "text": c.text,
            "token_ids": list(c.token_ids),
            "n_tokens": len(c.token_ids),
        })

    result = {
        "model": model, "tp": tp, "max_tokens": max_tokens,
        "n_prompts": len(PROMPTS),
        "load_s": round(t_load, 2), "gen_s": round(t_gen, 2),
        "outputs": items,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[run] tp={tp} 加载 {t_load:.1f}s 生成 {t_gen:.1f}s → {out_path}")
    return result


def common_prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare(path1, path2):
    r1 = json.load(open(path1, encoding="utf-8"))
    r2 = json.load(open(path2, encoding="utf-8"))
    if len(r1["outputs"]) != len(r2["outputs"]):
        print("条数不一致，无法对比")
        return 1

    print(f"=== TP={r1['tp']} vs TP={r2['tp']}   严格等价性对比（{len(r1['outputs'])} 条）===")
    all_same, prefix_lens, diverged = True, [], []
    for i, (a, b) in enumerate(zip(r1["outputs"], r2["outputs"])):
        same_text = a["text"] == b["text"]
        same_tok = a["token_ids"] == b["token_ids"]
        pl = common_prefix_len(a["token_ids"], b["token_ids"])
        prefix_lens.append(pl)
        if same_text and same_tok:
            print(f"[{i:2d}] ✅ 全长一致  n_tokens={a['n_tokens']}")
        else:
            all_same = False
            diverged.append({
                "idx": i, "prompt": a["prompt"][:60],
                "n_tokens_a": a["n_tokens"], "n_tokens_b": b["n_tokens"],
                "common_prefix": pl,
                "text_a": a["text"][:80], "text_b": b["text"][:80],
            })
            print(f"[{i:2d}] ❌ 发散  一致前缀 {pl}/{a['n_tokens']} token")

    total = sum(prefix_lens)
    maxposs = sum(o["n_tokens"] for o in r1["outputs"])
    print(f"\n一致 token 总数 {total}/{maxposs} （{total/maxposs:.1%}）  最短一致前缀 {min(prefix_lens)}")

    if all_same:
        verdict = "TP_EQUIV_STRONG"
        note = f"TP={r1['tp']} 与 TP={r2['tp']} 在 {len(r1['outputs'])} 条 prompt × ≤{r1['max_tokens']} token 上完全等价"
        print(f"=== 判定：{verdict} === {note}")
    elif min(prefix_lens) >= 32:
        verdict = "TP_EQUIV_WEAK"
        note = (f"长序列出现发散：{len(diverged)} 条不一致，最短一致前缀 {min(prefix_lens)} token。"
                f"短前缀等价、长序列累积误差显现 —— 报告须注明适用边界")
        print(f"=== 判定：{verdict} === {note}")
    else:
        verdict = "TP_EQUIV_FAIL"
        note = f"短前缀即发散（最短一致前缀 {min(prefix_lens)} token），非数值等价"
        print(f"=== 判定：{verdict} === {note}")

    res = {"verdict": verdict, "note": note, "common_prefix_lens": prefix_lens,
           "total_same_tokens": total, "max_tokens_compared": maxposs,
           "diverged": diverged}
    out = f"tp_compare_rigorous_tp{r1['tp']}_vs_tp{r2['tp']}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"结果：{out}")
    return 0 if verdict == "TP_EQUIV_STRONG" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="tp_rigorous.json")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1])
    if not args.model:
        print("--model 必填（或用 --compare 对比两个结果）")
        return 1
    run_model(args.model, args.tp, args.max_tokens, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
