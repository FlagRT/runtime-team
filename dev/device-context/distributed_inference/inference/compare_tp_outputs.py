#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
compare_tp_outputs.py — TP=1 vs TP=N 输出逐字对比（O3，A 线）
═══════════════════════════════════════════════════════════════════════════════

【目标】对照同事 B 线 TP 验证（TP=1/2 逐字一致），验证 A 线 TP 张量并行
  数值等价性：同一模型、同一 prompt、同一 seed、固定 max_tokens 下，
  TP=2（及 TP=4）输出必须与 TP=1 逐字完全一致。
  同时复现同事两条缺陷的对照实验：
    - 缺陷① bool×int 类型提升错误（TP=2 专属）→ 若存在，输出与 TP=1 不一致
    - 缺陷② flagcx 异步 all_reduce 无同步 → NaN（A 线 P2 已证不存在，这里复核）
【用法】
  python compare_tp_outputs.py <tp1_result.json> <tpn_result.json>
【判定】全部 prompt 文本 + token_ids 一致 → TP_COMPARE_PASS
"""

import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tp1", help="TP=1 的 dense_infer_result.json")
    ap.add_argument("tpn", help="TP=N 的 dense_infer_result.json")
    args = ap.parse_args()

    r1 = load(args.tp1)
    rn = load(args.tpn)

    print("=" * 70)
    print(f"TP=1 : {args.tp1}  (tp={r1.get('tp')}, model={r1.get('model')}, verdict={r1.get('verdict')})")
    print(f"TP=N : {args.tpn}  (tp={rn.get('tp')}, model={rn.get('model')}, verdict={rn.get('verdict')})")
    print("=" * 70)

    # 前置校验：模型一致、TP 不同、基线必须 PASS
    if r1.get("model") != rn.get("model"):
        print(f"TP_COMPARE_ABORT: 模型不一致 {r1.get('model')} vs {rn.get('model')}")
        sys.exit(1)
    if r1.get("tp") == rn.get("tp"):
        print(f"TP_COMPARE_ABORT: 两次运行 TP 相同（{r1.get('tp')}），无法对比张量并行")
        sys.exit(1)
    if r1.get("verdict") != "DENSE_INFER_PASS":
        print(f"TP_COMPARE_ABORT: TP=1 基线未 PASS（{r1.get('verdict')}），先修复基线")
        sys.exit(1)

    o1 = r1.get("outputs_full", [])
    on = rn.get("outputs_full", [])
    if len(o1) != len(on):
        print(f"TP_COMPARE_ABORT: 输出条数不一致 {len(o1)} vs {len(on)}")
        sys.exit(1)

    # 逐 prompt 对比：文本逐字 + token_ids
    n_pass = 0
    diffs = []
    for i, (a, b) in enumerate(zip(o1, on)):
        same_text = a["text"] == b["text"]
        same_tok = a["token_ids"] == b["token_ids"]
        if same_text and same_tok:
            n_pass += 1
            print(f"[{i}] ✅ 一致 (text+token_ids)  len={len(a['text'])}")
        else:
            diffs.append(i)
            print(f"[{i}] ❌ 不一致  text_same={same_text} tok_same={same_tok}")
            print(f"      TP=1: {a['text'][:100]!r}")
            print(f"      TP=N: {b['text'][:100]!r}")
            if a["token_ids"] != b["token_ids"]:
                la, lb = len(a["token_ids"]), len(b["token_ids"])
                print(f"      token_ids len {la} vs {lb}  首个分歧位置: "
                      f"{next((k for k in range(min(la, lb)) if a['token_ids'][k] != b['token_ids'][k]), '长度不同')}")

    # 性能对照（仅参考，不参与判定）
    t1 = r1.get("timing", {})
    tn = rn.get("timing", {})
    print("-" * 70)
    print(f"性能对照      TP=1          TP={rn.get('tp')}")
    print(f"  加载 (s)    {t1.get('load_s', '?'):>10}    {tn.get('load_s', '?'):>10}")
    print(f"  推理 (s)    {t1.get('infer_s', '?'):>10}    {tn.get('infer_s', '?'):>10}")
    print(f"  吞吐 tok/s  {t1.get('tok_s', '?'):>10}    {tn.get('tok_s', '?'):>10}")
    print(f"  tokens      {t1.get('total_tokens', '?'):>10}    {tn.get('total_tokens', '?'):>10}")

    # NaN 复核（缺陷② A 线对照）
    def has_nan(r):
        return any("nan" in o["text"].lower() for o in r.get("outputs_full", []))
    nan1, nanN = has_nan(r1), has_nan(rn)
    if nanN:
        print("\n⚠️ 检出 NaN！对照同事缺陷②（flagcx 异步无同步→NaN），A 线不应出现")
    else:
        print(f"\nNaN 复核: TP=1 {nan1} / TP=N {nanN}（缺陷② A 线对照：无 NaN）")

    verdict = "TP_COMPARE_PASS" if not diffs and n_pass == len(o1) else "TP_COMPARE_FAIL"
    print(f"\n{verdict}: {n_pass}/{len(o1)} 条逐字一致{'，差异条目: ' + str(diffs) if diffs else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
