"""快速诊断：XPU 上 embed 的确定性 / 批组成敏感性 / 语义合理性。"""
import numpy as np


def main() -> None:
    from vllm import LLM

    llm = LLM(
        model="/work/models/qwen3-embed-0.6b-xpu",
        task="embed",
        max_model_len=4096,
        dtype="float16",
        enforce_eager=True,
    )

    def E(texts):
        out = llm.embed(list(texts), use_tqdm=False)
        return np.array([np.asarray(o.outputs.embedding) for o in out])

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    # 1) 重复调用确定性：同文本单条调用两次
    v1 = E(["昆仑芯嵌入测试"])[0]
    v2 = E(["昆仑芯嵌入测试"])[0]
    print(f"[1] 单条重复调用: cos={cos(v1, v2):.6f} array_equal={np.array_equal(v1, v2)}")

    # 2) 批组成敏感性：同一文本分别在 bs=1 / 3条批第1位 / 3条批第3位
    others = ["完全不相关的填充文本甲", "另一段不相关的填充文本乙"]
    v_solo = E(["目标文本用于嵌入对齐检测"])[0]
    b3 = E([others[0], "目标文本用于嵌入对齐检测", others[1]])
    v_mid = b3[1]
    print(f"[2] 同文本 bs=1 vs bs=3中间位: cos={cos(v_solo, v_mid):.6f}")

    # 3) 语义 sanity：近义 vs 无关
    s = E(["一只猫在沙发上睡觉", "小猫趴在软垫上打盹", "季度财务报表分析"])
    print(f"[3] 语义: cos(近义)={cos(s[0], s[1]):.4f}  cos(无关)={cos(s[0], s[2]):.4f}")

    # 4) 短文本混合长度批（复现 bench 的 C 路径形态：分桶后同批长度接近）
    short = ["短文本" + "啊" * 5] * 2
    long_ = ["长文本" + "字" * 300] * 2
    mixed = E(short + long_)
    solo_short = E([short[0]])[0]
    print(f"[4] 长短混批 vs 单条(短): cos={cos(mixed[0], solo_short):.6f}")


if __name__ == "__main__":
    main()
