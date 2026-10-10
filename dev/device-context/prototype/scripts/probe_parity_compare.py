#!/usr/bin/env python3
"""四家「同名探针产物」逐项数值同比对（同口径深度核验）。

找到四家各自的同类结果 JSON，逐项列出 checks 的键与 ok 值，
回答「同一套探针在四家上跑出的项数与结论形状是否一致」。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]   # dev/device-context
CHIPS = ["910C", "P800", "MLU590", "PPU"]

# 探针 → 候选文件名（按优先级）
TARGETS = {
    "stream_semantics_full": ["stream_semantics_full_result.json", "stream_semantics_full_result_ppu.json"],
    "probe_bc_contract": ["probe_bc_contract_*.json", "m1_bc_probe_*.json"],
    "duty_audit": ["duty_audit_*.json", "d2_duty_*.json", "m1_duty*.json"],
    "graph_capture": ["graph_capture*result*.json", "graph_capture_stream_v2*.json"],
    "stream_quota": ["stream_quota*result*.json", "b2_stream_quota*.json"],
}


def newest(paths):
    paths = [p for p in paths if p.is_file()]
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None


def load(p):
    try:
        return json.loads(p.read_text(errors="replace"))
    except Exception:
        return None


def summarize(name, d):
    if not isinstance(d, dict):
        return "(无法解析)"
    keys = ("checks", "results", "rows", "items")
    for k in keys:
        v = d.get(k)
        if isinstance(v, dict):
            ok = sum(1 for x in v.values() if (x.get("ok") if isinstance(x, dict) else bool(x)))
            return f"{k}: {ok}/{len(v)} 项 ok | verdict={d.get('verdict') or d.get('PASS')}"
        if isinstance(v, list):
            ok = sum(1 for x in v if (x.get("ok") if isinstance(x, dict) else bool(x)))
            return f"{k}: {ok}/{len(v)} 项 ok | verdict={d.get('verdict') or d.get('PASS')}"
    flat = {k: v for k, v in d.items() if isinstance(v, bool)}
    if flat:
        return f"扁平布尔 {sum(flat.values())}/{len(flat)} | keys={sorted(flat)[:8]}"
    return f"keys={sorted(d)[:10]}"


def main():
    for probe, pats in TARGETS.items():
        print("=" * 96)
        print(f"探针: {probe}")
        print("=" * 96)
        for c in CHIPS:
            pdir = ROOT / c / "probes"
            found = []
            for pat in pats:
                found += list(pdir.rglob(pat))
            p = newest(found)
            if p is None:
                print(f"  {c:8s} ❌ 无产物")
                continue
            d = load(p)
            print(f"  {c:8s} ✅ {p.relative_to(pdir)}\n           {summarize(probe, d)}")
        print()


if __name__ == "__main__":
    main()
