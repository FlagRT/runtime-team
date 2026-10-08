#!/usr/bin/env python3
"""Resolve pins.<tag>.yaml layers matching the given docker-compose.<tag>.yml
files, check the declared image digest and repo versions against the host,
write a provenance snapshot, and print a PASS/WARN/FAIL report.

Schema and rules: see dev/ENV-SPEC.md.

Usage: verify_env.py <docker-compose-file1> [docker-compose-file2 ...]
Exit 0 iff no strict/image mismatch. loose mismatches and dirty working
trees are reported but never fail the check.
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

REPO_DIR = {
    "FlagTree": "FlagTree",
    "FlagGems": "FlagGems",
    "Torch-FL": "PyTorch-Plugin-FL",
    "FlagCX": "FlagCX",
    "vllm-plugin-FL": "vllm-plugin-FL",
    "FlagPerf": "FlagPerf",
}


def deep_merge(base: dict, overlay: dict) -> dict:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def pins_candidates_for(compose_path: Path) -> list:
    """derive candidate pins paths for a compose file: same dir, 'pins.' prefix
    replacing 'docker-compose.', trying both .yml/.yaml regardless of which
    extension the compose file itself uses (authors will mix the two)."""
    name = compose_path.name
    rest = name[len("docker-compose."):] if name.startswith("docker-compose.") else name
    stem = rest.rsplit(".", 1)[0] if "." in rest else rest
    return [
        compose_path.parent / f"pins.{stem}.yaml",
        compose_path.parent / f"pins.{stem}.yml",
    ]


def load_resolved_pins(compose_paths) -> dict:
    resolved: dict = {}
    for cp in compose_paths:
        candidates = pins_candidates_for(Path(cp))
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            print(f"[INFO] 无版本声明（未找到 {candidates[0].name} 或 {candidates[1].name}），跳过: {cp}")
            continue
        data = yaml.safe_load(found.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            print(f"[FAIL] pins 文件顶层不是映射: {found}", file=sys.stderr)
            sys.exit(2)
        resolved = deep_merge(resolved, data)
    return resolved


def run_git(*args, cwd):
    try:
        proc = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10
        )
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    if proc.returncode != 0:
        return None, proc.stderr.strip()
    return proc.stdout.strip(), None


def check_image(pins: dict, report: list) -> bool:
    image = pins.get("image")
    if not image:
        return True
    ref = image.get("ref")
    expect_digest = image.get("digest")
    if not ref or not expect_digest:
        report.append(("FAIL", "image", f"pins 声明了 image 但缺 ref/digest: {image}"))
        return False
    proc = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True, text=True, timeout=15,
    )
    if proc.returncode != 0:
        report.append(("FAIL", "image", f"本机没有该镜像或 ref 有误: {ref} ({proc.stderr.strip()})"))
        return False
    actual = proc.stdout.strip()
    if actual != expect_digest:
        report.append(("FAIL", "image", f"digest 不匹配: 期望 {expect_digest}, 实际 {actual} (ref={ref})"))
        return False
    report.append(("PASS", "image", f"{ref} -> {actual}"))
    return True


def check_repo(name: str, spec: dict, report: list) -> bool:
    dirname = REPO_DIR.get(name)
    if dirname is None:
        report.append(("FAIL", name, "未知仓库名，需在 verify_env.py 的 REPO_DIR 里登记"))
        return False
    path = REPO_ROOT / dirname
    cls = spec.get("class", "loose")

    if not path.is_dir():
        report.append(("FAIL" if cls == "strict" else "WARN", name, f"宿主路径不存在: {path}"))
        return cls != "strict"

    head, err = run_git("rev-parse", "HEAD", cwd=path)
    if head is None:
        report.append(("FAIL" if cls == "strict" else "WARN", name, f"不是合法 git 仓库: {err}"))
        return cls != "strict"

    ok = True
    if cls == "strict":
        expect_commit = spec.get("expect_commit")
        if not expect_commit:
            report.append(("FAIL", name, "class=strict 但未声明 expect_commit"))
            ok = False
        elif head != expect_commit:
            report.append(("FAIL", name, f"commit 不匹配: 期望 {expect_commit[:12]}, 实际 {head[:12]}"))
            ok = False
        else:
            report.append(("PASS", name, f"commit={head[:12]}"))
    else:
        branch, _ = run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=path)
        expect_branch = spec.get("expect_branch")
        if expect_branch and branch != expect_branch:
            report.append(("WARN", name, f"当前分支 {branch} != 期望 {expect_branch}（loose，不阻断）"))
        else:
            report.append(("PASS", name, f"commit={head[:12]} branch={branch}"))

    dirty_out, _ = run_git("status", "--porcelain", cwd=path)
    if dirty_out:
        report.append(("WARN", name, "working tree 有未提交改动"))

    return ok


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify_env.py <docker-compose-file...>", file=sys.stderr)
        return 2

    compose_paths = sys.argv[1:]
    pins = load_resolved_pins(compose_paths)

    report: list = []
    ok = True
    ok &= check_image(pins, report)
    for name, spec in (pins.get("repos") or {}).items():
        ok &= check_repo(name, spec, report)

    for status, name, msg in report:
        print(f"[{status}] {name}: {msg}")

    provenance = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "compose_files": compose_paths,
        "resolved_pins": pins,
        "report": [{"status": s, "item": n, "detail": m} for s, n, m in report],
        "result": "PASS" if ok else "FAIL",
    }
    out_path = Path(__file__).resolve().parent / ".env_provenance.json"
    out_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"provenance -> {out_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
