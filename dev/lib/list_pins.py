#!/usr/bin/env python3
"""Scan the repo for every pins.<tag>.yaml and show, per repo, what each
layer declares side by side -- for spotting where different candidate
lineages / exploration layers pin the same library to different commits.

Reads each pins file standalone (not merged across a -f stack): the point
here is to compare raw declarations across layers, not resolve one launch.

Usage:
  list_pins.py                    # every repo found in any pins file
  list_pins.py FlagCX FlagGems    # only these repos
  list_pins.py --images           # also list image ref/digest per file
"""
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_pins_files():
    return sorted(REPO_ROOT.glob("**/pins.*.y*ml"))


def rel(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT))


def main() -> int:
    args = sys.argv[1:]
    show_images = "--images" in args
    repo_filter = [a for a in args if not a.startswith("--")]

    files = find_pins_files()
    if not files:
        print("未找到任何 pins.*.yaml 文件")
        return 0

    by_repo: dict = {}
    images: list = []

    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            print(f"[跳过，顶层不是映射] {rel(f)}", file=sys.stderr)
            continue
        for name, spec in (data.get("repos") or {}).items():
            by_repo.setdefault(name, []).append((f, spec))
        if data.get("image"):
            images.append((f, data["image"]))

    repo_names = sorted(by_repo) if not repo_filter else [r for r in repo_filter if r in by_repo]
    if repo_filter:
        for m in repo_filter:
            if m not in by_repo:
                print(f"(未在任何 pins 文件中找到 {m} 的声明)")

    for name in repo_names:
        entries = by_repo[name]
        commits = {spec.get("expect_commit") for _, spec in entries if spec.get("class") == "strict"}
        commits.discard(None)
        flag = "  <-- 不同文件声明了不同 commit" if len(commits) > 1 else ""
        print(f"\n=== {name} ==={flag}")
        for f, spec in entries:
            cls = spec.get("class", "loose")
            if cls == "strict":
                detail = f"strict  expect_commit={spec.get('expect_commit', '<缺失>')}"
            else:
                detail = f"loose   expect_branch={spec.get('expect_branch', '<未声明>')}"
            print(f"  {rel(f):70s} {detail}")

    if show_images:
        print("\n=== image ===")
        for f, image in images:
            print(f"  {rel(f):70s} ref={image.get('ref')}")
            print(f"  {'':70s} digest={image.get('digest')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
