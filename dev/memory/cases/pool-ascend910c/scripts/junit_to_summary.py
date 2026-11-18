#!/usr/bin/env python3
"""pool-ascend910c：junit.xml → evidence/summary.json（机器可判结果）。

输入：pytest --junitxml 产物、env/v2 的 env.baseline.yaml、容器内 cann-shmem pip show 输出。
输出字段：result / counts / tests（用例名→pass|fail）/ image（ref+id）/ env（挂载 commit 锚
+ 层包实装版本）/ executed_at。判定口径：11 用例全部 pass 才 PASS（对齐样例判据）。

用法：junit_to_summary.py <junit.xml> <env.baseline.yaml> <out.json> <cann-shmem-pipshow.txt>
退出码：0 = 全部通过；1 = 有失败/无法判定。
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

EXPECT_TOTAL = 11   # tests/pool/test_pool_realcard.py 全量用例数（判据口径）


def main(argv):
    if len(argv) != 5:
        print(__doc__, file=sys.stderr)
        return 2
    junit, baseline_path, out_path, pipshow = argv[1:5]
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
    env_pkgs = {p.get("name"): p.get("version") for p in base.get("env_packages") or []}
    shmem_installed = None
    try:
        txt = Path(pipshow).read_text(encoding="utf-8")
        m = re.search(r"^Version: (.+)$", txt, re.M)
        shmem_installed = m.group(1).strip() if m else None
    except OSError:
        pass

    all_pass = (counts["total"] == EXPECT_TOTAL and counts["passed"] == EXPECT_TOTAL)
    summary = {
        "schema": "flagrt.case-summary/v1",
        "case": "pool-ascend910c",
        "executed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "result": "PASS" if all_pass else "FAIL",
        "criteria": {"suite": "FlagMemory tests/pool/test_pool_realcard.py",
                     "expect_total": EXPECT_TOTAL},
        "counts": counts,
        "tests": tests,
        "image": {"ref": img.get("ref"), "id": img.get("id")},
        "env": {"dir": "dev/images/ascend-train-comm/v3/env/v2",
                "mounted_repos": mounted,
                "env_packages": env_pkgs,
                "cann-shmem_installed": shmem_installed},
    }
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"summary -> {out_path}  result={summary['result']}  "
          f"counts={counts['passed']}/{counts['total']} passed"
          + (f" failed={counts['failed']}" if counts["failed"] else "")
          + (f" errors={counts['errors']}" if counts["errors"] else "")
          + (f" skipped={counts['skipped']}" if counts["skipped"] else ""))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
