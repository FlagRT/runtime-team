"""混批缺陷定位实验：先排除结果-请求对应错误，再测损坏跟随文本还是位置、长度差阈值。

实验设计（全部小规模，单卡空闲期运行）：
  E1 映射判别: 2短+2长混合批 vs 各自单条参照，打印 4x4 交叉余弦矩阵
     - 对角线≈1  → 映射正确，问题在数值计算（pooling/attention）
     - 非对角≈1  → 结果错位/置换（映射错误）
     - 全无≈1    → 向量本身算错（不对应任何输入）
  E2 顺序交换: [短,长] vs [长,短]，看损坏跟随文本还是槽位
  E3 长度差阈值: 固定短文本(≈10 tok)，长侧从 16→256 扫描，找 cos 崩坏的临界
  E4 长长小差: ≈300 vs ≈320 tok，判断是"长度差"还是"绝对短"
"""

import json
import sys

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
        """构造 ≈target token 的文本，内容按 seed_word 区分。"""
        words = [seed_word] * (target + 8)
        text = " ".join(words)
        ids = tok.encode(text)
        while len(ids) > target:  # 逐词收缩到预算内
            words = words[:-1]
            text = " ".join(words)
            ids = tok.encode(text)
        return text, len(ids)

    # 可信参照：全部单条调用（已证单条完全确定）
    texts = [
        make(10, "甲甲")[0],     # 短A
        make(10, "乙乙")[0],     # 短B
        make(300, "丙丙")[0],    # 长A
        make(300, "丁丁")[0],    # 长B
    ]
    lens = [len(tok.encode(t)) for t in texts]
    refs = [E([t])[0] for t in texts]

    # ---- E1: 映射判别 ----
    V = E(texts)
    print("\n[E1] 4x4 交叉余弦矩阵（行=混批返回第i个，列=单条参照第j条）")
    print("     token长度:", lens)
    header = "        " + "".join(f"  ref{j}({lens[j]}tok)" for j in range(4))
    print(header)
    for i in range(4):
        row = "  ".join(f"{cos(V[i], refs[j]):+8.4f}" for j in range(4))
        print(f"  out{i}   {row}")

    # ---- E2: 顺序交换 ----
    s, l = texts[0], texts[2]
    ref_s, ref_l = refs[0], refs[2]
    a1 = E([s, l])  # [短, 长]
    a2 = E([l, s])  # [长, 短]
    print("\n[E2] 顺序交换：cos(返回值, 对应参照)")
    print(f"  [短,长]: 短位cos={cos(a1[0], ref_s):+.4f}  长位cos={cos(a1[1], ref_l):+.4f}")
    print(f"  [长,短]: 长位cos={cos(a2[0], ref_l):+.4f}  短位cos={cos(a2[1], ref_s):+.4f}")

    # ---- E3: 长度差阈值 ----
    print("\n[E3] 固定短(≈10tok)，扫描长侧长度，短文本结果 cos（1.0=无损）")
    s_text, _ = make(10, "戊戊")
    ref = E([s_text])[0]
    e3 = []
    for L in (16, 24, 32, 48, 64, 96, 128, 192, 256):
        lt, actual = make(L, "己己")
        pair = E([s_text, lt])
        c = cos(pair[0], ref)
        e3.append({"long_target": L, "long_actual": actual, "cos": round(c, 6)})
        print(f"  短10 + 长{actual:>3}: cos={c:+.4f}")

    # ---- E4: 长长小差 ----
    t300, n300 = make(300, "庚庚")
    t320, n320 = make(320, "辛辛")
    r300 = E([t300])[0]
    r320 = E([t320])[0]
    pair = E([t300, t320])
    print("\n[E4] 长长小差（≈300 vs ≈320 tok）")
    print(f"  300位 cos={cos(pair[0], r300):+.4f}   320位 cos={cos(pair[1], r320):+.4f}")

    # 保存
    out = {
        "e1_matrix": [[round(cos(V[i], refs[j]), 6) for j in range(4)] for i in range(4)],
        "e1_token_lens": lens,
        "e2": {"short_long": [round(cos(a1[0], ref_s), 6), round(cos(a1[1], ref_l), 6)],
                "long_short": [round(cos(a2[0], ref_l), 6), round(cos(a2[1], ref_s), 6)]},
        "e3_threshold": e3,
        "e4": [round(cos(pair[0], r300), 6), round(cos(pair[1], r320), 6)],
    }
    with open("/work/ctools/mapping_probe_result.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 /work/ctools/mapping_probe_result.json")


if __name__ == "__main__":
    main()
