#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_error_map_coverage.py — D10 覆盖率与分级差异审计

【做什么】把 CANN 头文件提取的 132 个错误码，逐个构造错误消息喂给当前
         conformance/errors.py 的 translate_error，得到「当前分级」，
         与 gen_acl_error_map.py 给出的「建议分级」对比，量化改进空间。

【为什么】错误码翻译的验证不必依赖逐个真实触发。分三层：
          L1 覆盖审计（本脚本，头文件 → 映射表覆盖率与分级差异）
          L2 翻译链路验证（构造消息 → 分级是否符合预期，可批量自动化）
          L3 真实触发验证（抽样，如 107015 的 A/B 对照）
         本脚本覆盖 L1+L2，是 D10 可规模化的验证手段。

【用法】容器内：python3 audit_error_map_coverage.py [--candidate acl_error_map_candidate.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONF_DIR = os.path.abspath(os.path.join(HERE, "..", "ascend_regression", "conformance"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="acl_error_map_candidate.json")
    ap.add_argument("--out", default="acl_error_map_audit.json")
    args = ap.parse_args()

    sys.path.insert(0, CONF_DIR)
    from errors import translate_error, ACL_ERR_TO_CATEGORY  # noqa: E402

    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)

    current_dist, suggested_dist = {}, {}
    diffs, in_map = [], []

    for e in cand["entries"]:
        code = e["code"]
        msg = f"ACL runtime error: {e['meaning']}, error code is {code}"
        fe = translate_error(RuntimeError(msg), location=f"device:0/op:{e['name']}")
        cur, sug = fe.category.name, e["suggested_category"]
        current_dist[cur] = current_dist.get(cur, 0) + 1
        suggested_dist[sug] = suggested_dist.get(sug, 0) + 1
        if code in ACL_ERR_TO_CATEGORY:
            in_map.append(code)
        if cur != sug:
            diffs.append({
                "code": code,
                "name": e["name"],
                "meaning": e["meaning"],
                "current": cur,
                "suggested": sug,
                "confidence": e["confidence"],
            })

    total = len(cand["entries"])
    result = {
        "total_codes": total,
        "in_current_map": len(in_map),
        "coverage_pct": round(100.0 * len(in_map) / total, 1),
        "current_distribution": current_dist,
        "suggested_distribution": suggested_dist,
        "diff_count": len(diffs),
        "diff_pct": round(100.0 * len(diffs) / total, 1),
        "diffs_high_confidence": [
            d for d in diffs if d["confidence"] == "high"
        ][:40],
        "diffs_sample": diffs[:20],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"=== D10 错误码翻译审计（{total} 个 CANN 错误码）===")
    print(f"当前映射表覆盖: {len(in_map)}/{total} = {result['coverage_pct']}%")
    print(f"当前分级分布: {current_dist}")
    print(f"建议分级分布: {suggested_dist}")
    print(f"分级不一致: {len(diffs)} 个 ({result['diff_pct']}%)")
    print(f"  其中高置信建议的差异: {len(result['diffs_high_confidence'])} 个")
    print("--- 高置信差异样例（最值得优先修正）---")
    for d in result["diffs_high_confidence"][:12]:
        print(f"  {d['code']} {d['name'][:38]:38s} {d['current']:14s} → {d['suggested']}")
    print(f"结果：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
