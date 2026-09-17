#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
dense_infer_qwen_1_5b_npu.py — 同构 910C dense 单卡离线推理（P0，A 线）
═══════════════════════════════════════════════════════════════════════════════

【目标】验证 A 线推理栈（torch_npu + vLLM + vllm-plugin-FL）在 910C 上
  dense 单卡离线推理可用，作为设备上下文职责（P1/P2）的基线环境。
【用法】容器内、A 线 venv（含 vLLM 0.20.2 + vllm-plugin-FL）：
  export DO_NOT_TRACK=1 VLLM_PLUGINS=fl
  python dense_infer_qwen_1_5b_npu.py --model <path> [--preheat]
【输出】stdout + dense_infer_result.json：加载耗时 / 首 token 延迟 / 吞吐 tok/s
【纪律】
  - 预热（坑 A1）：首次 attention 可能 13+ 分钟，--preheat 先跑短请求再计时
  - 官方发布镜像内执行（坑 A4），不用 dev 容器做结论性测试
  - 判定：输出非空、无 NaN/乱码
"""

import argparse
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型路径或 HF 名（如 /root/models/Qwen2.5-1.5B）")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size（TP=1/2/4，O3 对比用）")
    ap.add_argument("--seed", type=int, default=42, help="采样随机种子（TP 对比需固定以保证可复现）")
    ap.add_argument("--greedy", action="store_true",
                    help="greedy 解码（temperature=0，确定性，TP 逐字对比的正确对照方式；"
                         "随机采样下 TP 浮点累加顺序差异会被放大导致必然不一致）")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--preheat", action="store_true", help="先跑 1 个短请求预热再计时（坑 A1）")
    ap.add_argument("--out", default="dense_infer_result.json")
    args = ap.parse_args()

    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "2+2=",
        "Python is a",
    ]

    result = {"verdict": "FAIL", "env": {}, "checks": {}, "timing": {}, "note": "",
              "model": args.model, "tp": args.tp, "seed": args.seed, "greedy": args.greedy}
    try:
        import torch, torch_npu
        result["env"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__}
        import vllm
        result["env"]["vllm"] = vllm.__version__
        result["env"]["VLLM_PLUGINS"] = os.environ.get("VLLM_PLUGINS", "(未设置)")
        print(f"[env] torch={torch.__version__} torch_npu={torch_npu.__version__} "
              f"vllm={vllm.__version__} plugins={os.environ.get('VLLM_PLUGINS')}")
    except ImportError as e:
        print(f"DENSE_INFER_ABORT: 组件缺失 {e}（需 vLLM 0.20.2 + vllm-plugin-FL，见方案 P0 环境准备）")
        result["note"] = f"组件缺失: {e}"
        _dump(result, args.out)
        return

    from vllm import LLM, SamplingParams

    # ── 加载（计时）──
    t0 = time.time()
    print(f"[load] 加载模型 {args.model} (tp={args.tp}) ...")
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, enforce_eager=True)
    t_load = time.time() - t0
    result["timing"]["load_s"] = round(t_load, 2)
    print(f"[load] 完成 {t_load:.2f}s")

    # ── 预热（坑 A1：首次 attention 极慢）──
    if args.preheat:
        print("[preheat] 先跑 1 个短请求预热（首次 attention 可能 13+ 分钟）...")
        llm.generate(["Hi"], SamplingParams(max_tokens=8, temperature=0.0 if args.greedy else 1.0, seed=args.seed))
        print("[preheat] 完成")

    # ── 正式推理（计时）──
    params = SamplingParams(max_tokens=args.max_tokens,
                            temperature=0.0 if args.greedy else 1.0,
                            seed=args.seed)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    t_infer = time.time() - t0
    n_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    tok_s = n_tokens / t_infer if t_infer > 0 else 0
    result["timing"]["infer_s"] = round(t_infer, 2)
    result["timing"]["total_tokens"] = n_tokens
    result["timing"]["tok_s"] = round(tok_s, 2)

    # ── 判定 ──
    texts = [o.outputs[0].text for o in outs]
    ok = all(len(t.strip()) > 0 for t in texts) and not any("nan" in t.lower() for t in texts)
    result["checks"]["outputs"] = [t[:80] for t in texts]
    # 完整输出与 token ids（供 O3 逐字对比脚本使用，避免截断造成误判）
    result["outputs_full"] = [{"text": t, "token_ids": o.outputs[0].token_ids} for t, o in zip(texts, outs)]
    result["verdict"] = "DENSE_INFER_PASS" if ok else "DENSE_INFER_FAIL"
    result["note"] = f"加载 {t_load:.1f}s / 推理 {t_infer:.1f}s / {n_tokens} tokens / {tok_s:.1f} tok/s"
    print(f"\n{result['verdict']}: {result['note']}")
    for p, t in zip(prompts, texts):
        print(f"  [{p[:20]!r}] → {t[:60]!r}")

    _dump(result, args.out)


def _dump(result, out):
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
