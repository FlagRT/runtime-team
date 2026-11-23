"""score() 入口两缺陷测试：bge-reranker-base（XLMRoberta，CLS 池化）。

Bug1（缓存命中）: 缓存开，同一长文本对连打两次分 → 分数是否漂移
Bug2（跨步）: 缓存关，40对×~450tok（总~18000>16384，2步）批量 vs 逐对单独
"""
import argparse

MODEL = "/data1/flagperf-kzhang519/p800-models/bge-reranker-base"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=("bug1", "bug2"), required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM

    tok = AutoTokenizer.from_pretrained(MODEL)

    def make_pair(seed_q, seed_d, d_target=440):
        q = " ".join([seed_q] * 25)
        words = [seed_d] * (d_target + 8)
        d = " ".join(words)
        while len(tok.encode(q + d)) > 500:
            words = words[:-1]
            d = " ".join(words)
        return q, d

    def S(llm, qs, ds):
        out = llm.score(list(qs), list(ds))
        vals = []
        for o in out:
            v = getattr(o.outputs, "score", None)
            if v is None:
                v = float(o.outputs.data)
            vals.append(float(v))
        return vals

    if args.part == "bug1":
        llm = LLM(model=MODEL, task="score", max_model_len=512,
                  dtype="float32", enforce_eager=True)  # 缓存默认开
        q, d = make_pair("问", "答")
        n = len(tok.encode(q + d))
        s1 = S(llm, [q], [d])[0]
        s2 = S(llm, [q], [d])[0]  # 第二次全量命中
        print(f"[score-Bug1] 对长~{n}tok: 第一次={s1:.6f} 第二次={s2:.6f} "
              f"漂移={abs(s1 - s2):.2e} {'❌中招' if abs(s1 - s2) > 1e-3 else '✅一致'}")
    else:
        llm = LLM(model=MODEL, task="score", max_model_len=512,
                  dtype="float32", enforce_eager=True,
                  enable_prefix_caching=False)
        seeds = ["甲乙", "丙丁", "戊己", "庚辛", "壬癸", "子丑", "寅卯", "辰巳",
                 "午未", "申酉", "戌亥", "元亨", "利贞", "天地", "玄黄", "宇宙",
                 "洪荒", "日月", "盈昃", "辰宿", "列张", "寒来", "暑往", "秋收",
                 "冬藏", "闰余", "成岁", "律吕", "调阳", "云腾", "致雨", "露结",
                 "为霜", "金生", "丽水", "玉出", "昆冈", "剑号", "巨阙", "珠称"]
        pairs = [make_pair(f"问{i}", s) for i, s in enumerate(seeds)]
        qs = [q for q, _ in pairs]
        ds = [d for _, d in pairs]
        total = sum(len(tok.encode(q + d)) for q, d in pairs)
        solo = [S(llm, [q], [d])[0] for q, d in pairs]
        batch = S(llm, qs, ds)
        bad = [i for i in range(len(pairs)) if abs(batch[i] - solo[i]) > 1e-3]
        print(f"[score-Bug2] 40对总~{total}tok(≈2步): 损坏 {len(bad)}/40 {bad[:12]}"
              f"{' ← 步2全中' if bad and min(bad) >= 20 else ''}")
        if bad:
            i = bad[0]
            print(f"  样本: 单独={solo[i]:.6f} 批量={batch[i]:.6f}")


if __name__ == "__main__":
    main()
