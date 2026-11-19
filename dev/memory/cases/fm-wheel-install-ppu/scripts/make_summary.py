#!/usr/bin/env python3
"""fm-wheel-install-ppu：汇总三判据写 evidence/summary.json（机器可判，禁手改）。

用法：make_summary.py <out.json> <image_id> <wheel_sha256> <c1> [failures...]
c1 ∈ {pass,fail}（环境判据）；C2/C3 的失败项在 failures 里按名匹配。
"""
import datetime
import json
import sys

out, image_id, wheel_sha, c1 = sys.argv[1:5]
fails = sys.argv[5:]
cases = {
    "C1-env-strict": c1,
    "C2-install-face": "fail" if "install-face" in fails or "pip-install" in fails or "wheel-cp" in fails else "pass",
    "C3-smoke-e2e": "fail" if "smoke" in fails else "pass",
}
n_pass = sum(v == "pass" for v in cases.values())
summary = {
    "schema": "flagrt.case-summary/v1",
    "case": "fm-wheel-install-ppu",
    "generated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "result": "PASS" if n_pass == 3 else "FAIL",
    "counts": {"pass": n_pass, "fail": 3 - n_pass, "skip": 0},
    "cases": cases,
    "image": {"id": image_id},
    "wheel": {"name": "flagmemory-0.3.0-py3-none-any.whl", "sha256": wheel_sha},
}
with open(out, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
print(f"[summary] {summary['result']} ({n_pass}/3)")
