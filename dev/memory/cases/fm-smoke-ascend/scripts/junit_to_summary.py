#!/usr/bin/env python3
"""fm-smoke-ascend：junit.xml → evidence/summary.json（机器可判结果，schema flagrt.case-summary/v1 简化）。

用法：junit_to_summary.py <junit.xml> <out.json> <image_id> <fm_commit>
只依赖标准库；junit 缺失时退出非 0（run.sh 如实记 WARN，不虚构结果）。
"""
import datetime
import json
import sys
import xml.etree.ElementTree as ET

junit, out, image_id, fm_commit = sys.argv[1:5]
root = ET.parse(junit).getroot()
cases = {}
for tc in root.iter("testcase"):
    name = (tc.get("name") or "?").split("[")[0].strip()
    if tc.find("failure") is not None or tc.find("error") is not None:
        cases[name] = "fail"
    elif tc.find("skipped") is not None:
        cases[name] = "skip"
    else:
        cases[name] = "pass"
n_pass = sum(v == "pass" for v in cases.values())
n_fail = sum(v == "fail" for v in cases.values())
n_skip = sum(v == "skip" for v in cases.values())
result = "PASS" if n_fail == 0 and n_skip == 0 and n_pass >= 3 else "FAIL"
summary = {
    "schema": "flagrt.case-summary/v1",
    "case": "fm-smoke-ascend",
    "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "result": result,
    "counts": {"pass": n_pass, "fail": n_fail, "skip": n_skip},
    "cases": cases,
    "image": {"id": image_id},
    "flagmemory_commit": fm_commit,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"[summary] {result} ({n_pass} pass / {n_fail} fail / {n_skip} skip)")
