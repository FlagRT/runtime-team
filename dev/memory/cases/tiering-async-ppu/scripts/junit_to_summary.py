#!/usr/bin/env python3
"""tiering-async-ppu：junit.xml → evidence/summary.json（机器可判结果）。

输入：pytest --junitxml 产物、case env 的 env.baseline.yaml。
判定口径：tests/tiering/test_ppu_tier_manager_realcard.py 全量 3 用例全部 pass 才 PASS。

用法：junit_to_summary.py <junit.xml> <env.baseline.yaml> <out.json>
退出码：0 = 全部通过；1 = 有失败/无法判定。
"""
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

EXPECT_TOTAL = 3   # test_ppu_tier_manager_realcard.py 全量用例数（T-8 负例/异步轨全链/destroy 排空）


def main(argv):
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    junit, baseline_path, out_path = argv[1:4]
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find(".//testsuite")
    tests = {}
    counts = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for case in suite.iter("testcase"):
        name = case.get("classname", "") + "::" + case.get("name", "")
        status = "pass"
        if case.find("failure") is not None:
            status = "fail"; counts["failed"] += 1
        elif case.find("error") is not None:
            status = "fail"; counts["errors"] += 1
        elif case.find("skipped") is not None:
            status = "skip"; counts["skipped"] += 1
        else:
            counts["passed"] += 1
        counts["total"] += 1
        tests[name] = status

    base = yaml.safe_load(Path(baseline_path).read_text(encoding="utf-8"))
    img = base.get("image") or {}
    mounted = {n: (i or {}).get("commit") for n, i in (base.get("mounted_repos") or {}).items()}
    all_pass = (counts["total"] == EXPECT_TOTAL and counts["passed"] == EXPECT_TOTAL)
    summary = {
        "schema": "flagrt.case-summary/v1",
        "case": "tiering-async-ppu",
        "executed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "result": "PASS" if all_pass else "FAIL",
        "criteria": {"suite": "FlagMemory tests/tiering/test_ppu_tier_manager_realcard.py",
                     "expect_total": EXPECT_TOTAL},
        "counts": counts,
        "tests": tests,
        "image": {"ref": img.get("ref"), "id": img.get("id")},
        "env": {"dir": "dev/memory/cases/tiering-async-ppu/env",
                "mounted_repos": mounted},
    }
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"summary -> {out_path}  result={summary['result']}  counts={counts['passed']}/{counts['total']} passed")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
