#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_acl_error_map.py — 从 CANN 头文件提取 ACL 错误码全集并按规则建议分级

【背景】D10 错误码翻译的瓶颈不是"没有错误示例"，而是没找到权威数据源。
        CANN 头文件即权威全集：
          /usr/local/Ascend/ascend-toolkit/latest/include/acl/error_codes/rt_error_codes.h（134 个）
          .../ge_error_codes.h（5 个）
        每条含 宏名 + 码值 + 英文语义注释 → 可直接作为映射表数据源与归类依据。

【产出】JSON：每条错误码的 {code, name, meaning, suggested_category, matched_rules, confidence}
        + 覆盖率/分级分布统计 + 需人工裁决的低置信清单

【分级规则（有序，先命中先得；多命中标记 low 置信度待人工裁决）】
  L4_FATAL    : 设备/上下文致命（device lost / context corrupt / fatal / crash / reset / aicore fault）
  L1_RESOURCE : 资源不足可重试（out of memory / alloc failed / resource busy / queue full）
  L2_PARAM    : 参数或契约类（invalid / param / unaligned / not support / format / null / exceed）
  L3_EXECUTION: 执行期（stream / event / task / execute / launch / sync / kernel / timestamp）兜底

【用法】容器内：python3 gen_acl_error_map.py [--out acl_error_map_candidate.json]
"""
import argparse
import json
import os
import re

ERR_DIR = "/usr/local/Ascend/ascend-toolkit/latest/include/acl/error_codes"
DEF_RE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+(\d+)\s*//\s*(.*?)\s*$")
SKIP_NAMES = ("__INC", "_H__", "ACL_RT_SUCCESS")

# 有序规则：(类别, 关键词元组)
RULES = [
    ("L4_FATAL", (
        "device lost", "no device", "device not", "context corrupt", "fatal",
        "crash", "reset", "aicore fault", "aicore exception", "hang",
        "unrecoverable", "dead",
    )),
    ("L1_RESOURCE", (
        "out of memory", "no memory", "memory alloc", "alloc failed", "alloc memory",
        "resource", "busy", "queue full", "no space", "exceed limit", "limit exceeded",
        "insufficient", "threshold",
    )),
    ("L2_PARAM", (
        "invalid", "param", "unaligned", "not support", "unsupport", "datatype",
        "format", "mismatch", "null", "exceed", "overflow", "illegal", "bad",
        "not registered", "not register", "not in current", "not in model",
        "uninitialized", "not init", "no cb", "not exist", "not found", "forbidden",
    )),
    # 收紧：去掉 fail/error/fault 等泛化词（几乎每条都含，导致兜底过宽、分布失真）
    ("L3_EXECUTION", (
        "stream", "event", "task", "execute", "exec ", "launch", "sync",
        "kernel", "timestamp", "internal error", "callback", "timeout",
        "not ready", "in progress", "abort",
    )),
]


def parse_header(path):
    """解析头文件 → [(name, code, meaning)]"""
    items = []
    if not os.path.exists(path):
        return items
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = DEF_RE.match(line)
            if not m:
                continue
            name, code, meaning = m.group(1), int(m.group(2)), m.group(3)
            if any(name.startswith(s) for s in SKIP_NAMES):
                continue
            # 跳过头文件保护宏等噪声
            if name.endswith("_H__") or name.startswith("__"):
                continue
            items.append((name, code, meaning))
    return items


def classify(meaning, name):
    """按有序规则建议分级；返回 (category, matched_rules, confidence)"""
    text = f"{meaning} {name}".lower()
    matched = []
    for cat, kws in RULES:
        hits = [k for k in kws if k in text]
        if hits:
            matched.append({"category": cat, "hits": hits})
    if not matched:
        # 无关键词可依：兜底 L3，但标 default 以便与"有依据的高/低置信"区分审计
        return "L3_EXECUTION", [], "default"
    if len(matched) == 1:
        return matched[0]["category"], matched, "high"
    # 多规则命中：取顺序最靠前的（规则已按严重度/精确度排序），但标 low
    return matched[0]["category"], matched, "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="acl_error_map_candidate.json")
    ap.add_argument("--dir", default=ERR_DIR)
    args = ap.parse_args()

    entries = []
    seen = set()
    for fn in sorted(os.listdir(args.dir)):
        if not fn.endswith(".h"):
            continue
        for name, code, meaning in parse_header(os.path.join(args.dir, fn)):
            if code in seen:
                continue
            seen.add(code)
            cat, matched, conf = classify(meaning, name)
            entries.append({
                "code": code,
                "name": name,
                "meaning": meaning,
                "header": fn,
                "suggested_category": cat,
                "confidence": conf,
                "matched": [
                    {"category": m["category"], "hits": m["hits"][:3]} for m in matched
                ],
            })

    entries.sort(key=lambda e: e["code"])
    stats = {
        "total": len(entries),
        "by_category": {},
        "by_confidence": {},
        "by_header": {},
    }
    for e in entries:
        stats["by_category"][e["suggested_category"]] = (
            stats["by_category"].get(e["suggested_category"], 0) + 1
        )
        stats["by_confidence"][e["confidence"]] = (
            stats["by_confidence"].get(e["confidence"], 0) + 1
        )
        stats["by_header"][e["header"]] = stats["by_header"].get(e["header"], 0) + 1

    low = [e for e in entries if e["confidence"] == "low"]

    result = {
        "source_dir": args.dir,
        "stats": stats,
        "need_manual_review": {
            "count": len(low),
            "sample": [
                {"code": e["code"], "name": e["name"], "meaning": e["meaning"],
                 "suggested": e["suggested_category"],
                 "conflict": [m["category"] for m in e["matched"]]}
                for e in low[:30]
            ],
        },
        "entries": entries,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"=== 提取完成：{stats['total']} 个错误码 ===")
    print(f"分级分布: {stats['by_category']}")
    print(f"置信度分布: {stats['by_confidence']}")
    print(f"按文件: {stats['by_header']}")
    print(f"需人工裁决（多规则冲突）: {len(low)} 个")
    if low:
        print("--- 冲突样例 ---")
        for e in low[:10]:
            cats = [m["category"] for m in e["matched"]]
            print(f"  {e['code']} {e['name']}: {e['meaning'][:44]!r} → {e['suggested_category']} {cats}")
    print(f"结果：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
