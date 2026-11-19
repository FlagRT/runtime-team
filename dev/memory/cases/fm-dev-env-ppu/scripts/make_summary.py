#!/usr/bin/env python3
"""fm-dev-env-ppu：汇总判据写 evidence/summary.json（机器可判，禁手改）。

用法：make_summary.py <out.json> <image_id> <fm_commit> <c1> [failures...]
c1 ∈ {pass,fail}（环境判据）；C2/C3 的失败项在 failures 里按名匹配。
"""
import datetime
import json
import sys

out, image_id, fm_commit, c1 = sys.argv[1:5]
fails = sys.argv[5:]
cases = {
    "C1-env-strict": c1,
    "C2-fm-anchor": "fail" if "fm-anchor" in fails else "pass",
    "C3a-backend-suite": "fail" if "suite-backend" in fails else "pass",
    "C3b-manager-suite": "fail" if "suite-manager" in fails else "pass",
}
n_pass = sum(v == "pass" for v in cases.values())
summary = {
    "schema": "flagrt.case-summary/v1",
    "case": "fm-dev-env-ppu",
    "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "result": "PASS" if n_pass == 4 else "FAIL",
    "counts": {"pass": n_pass, "fail": 4 - n_pass, "skip": 0},
    "cases": cases,
    "image": {"id": image_id},
    "fm_commit": fm_commit,
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"[summary] {summary['result']} ({n_pass}/4)")
