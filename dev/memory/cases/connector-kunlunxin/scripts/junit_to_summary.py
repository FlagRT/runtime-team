#!/usr/bin/env python3
"""connector-kunlunxin：junit.xml（×3）→ evidence/summary.json（机器可判结果）。

输入：三件判据的 pytest --junitxml 产物（offload/backend/tiermgr）、env/v1 的
env.baseline.yaml、容器内 fm-version.txt。
输出字段：result / counts / tests（用例名→pass|fail）/ image（ref+id）/ env（FM commit 锚
+ 容器内实装版本）/ executed_at。判定口径：判据三件共 6 用例全部 pass 且容器内
flagmemory == 0.3.0 才 PASS（对齐样例判据与 RUNBOOK）。

用法：junit_to_summary.py <env.baseline.yaml> <out.json> <fm-version.txt> <junit1> [<junit2> ...]
退出码：0 = 全部通过；1 = 有失败/无法判定；2 = 用法错误。
"""
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml

EXPECT_TOTAL = 5      # 判据三件用例数：offload(-k xpu) 1 + backend 1 + tier_manager 3（2026-11-15 真机实测收集数）
EXPECT_FM = "0.3.0"   # FlagMemory 发版版号（挂载锚 58e60a9 的 tag）


def main(argv):
    if len(argv) < 5:
        print(__doc__, file=sys.stderr)
        return 2
    baseline_path, out_path, fmver_path = argv[1:4]
    junit_files = argv[4:]
    tests = {}
    counts = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for jf in junit_files:
        root = ET.parse(jf).getroot()
        suite = root if root.tag == "testsuite" else root.find(".//testsuite")
        for case in suite.iter("testcase"):
            name = case.get("classname", "").rsplit(".", 1)[-1] + "::" + case.get("name", "")
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

    fm_installed = None
    try:
        txt = Path(fmver_path).read_text(encoding="utf-8")
        for ln in txt.splitlines():
            if ln.startswith("FM_VERSION"):
                fm_installed = ln.split()[-1].strip()
    except OSError:
        pass

    base = yaml.safe_load(Path(baseline_path).read_text(encoding="utf-8"))
    img = base.get("image") or {}
    mounted = {n: (i or {}).get("commit") for n, i in (base.get("mounted_repos") or {}).items()}

    all_pass = (counts["total"] == EXPECT_TOTAL and counts["passed"] == EXPECT_TOTAL
                and fm_installed == EXPECT_FM)
    summary = {
        "schema": "flagrt.case-summary/v1",
        "case": "connector-kunlunxin",
        "executed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "result": "PASS" if all_pass else "FAIL",
        "criteria": {
            "suite": "FlagMemory tests/tiering/ 判据三件（xpu RUNBOOK 口径）: "
                     "test_offload_realcard.py -k xpu / test_xpu_backend_realcard.py / "
                     "test_xpu_tier_manager_realcard.py",
            "expect_total": EXPECT_TOTAL,
            "expect_flagmemory": EXPECT_FM,
        },
        "counts": counts,
        "tests": tests,
        "image": {"ref": img.get("ref"), "id": img.get("id")},
        "env": {"dir": "dev/images/kunlun-infer-vllm/v1/env/v1",
                "mounted_repos": mounted,
                "flagmemory_installed": fm_installed},
    }
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"summary -> {out_path}  result={summary['result']}  "
          f"counts={counts['passed']}/{counts['total']} passed"
          + (f" failed={counts['failed']}" if counts["failed"] else "")
          + (f" errors={counts['errors']}" if counts["errors"] else "")
          + (f" skipped={counts['skipped']}" if counts["skipped"] else "")
          + (f"  flagmemory={fm_installed}"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
