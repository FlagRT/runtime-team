#!/usr/bin/env python3
"""机读锚点（RELEASE_INDEX.yaml）互锁校验：四查，红/绿输出。

① 逐字一致：每环境 index 的 image.ref / id / digest 与对应 image.baseline.yaml 逐字一致；
   环境键集合与七环境声明表（release_index.ENVIRONMENTS）一致，env_form / stack_lock /
   env_current.dir 与声明表一致。
② 锚点互见：index 中全部 id / digest 的前 8 位十六进制须在 USAGE_MATRIX.md（人读锚点）
   文本中出现；环境带 digest 锚而 id 未在矩阵展示时记 INFO（矩阵对 registry 锚环境的
   展示口径 = digest）。
③ 存在性：index 引用的 stack_lock、image.baseline 与 env_current 目录 / env.baseline.yaml 均存在。
④ 自证：generated_by 含 release_index.py，schema == flagrt.release-index/v1。

用法：check_release.py
退出码：0 全绿；1 存在 FAIL；2 文件缺失/格式错误。
"""
import sys
from pathlib import Path

import yaml

from release_index import ENVIRONMENTS

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "dev/images/RELEASE_INDEX.yaml"
MATRIX_PATH = ROOT / "dev/images/USAGE_MATRIX.md"
SCHEMA = "flagrt.release-index/v1"

PASS, FAIL, INFO = "[PASS]", "[FAIL]", "[INFO]"
fails = 0
infos = 0


def report(level: str, msg: str) -> None:
    global fails, infos
    print(f"{level} {msg}")
    if level == FAIL:
        fails += 1
    elif level == INFO:
        infos += 1


def anchor_prefix(value: str) -> str:
    """取 id/digest 串的前 8 位十六进制（人读锚点的展示口径）。"""
    hex_part = value.split("sha256:")[-1]
    return hex_part[:8].lower()


def check(index: dict, matrix_text: str) -> None:
    # ── ① index ↔ image.baseline.yaml 逐字一致 + 环境集合/形态与声明表一致 ──
    envs = index.get("environments") or {}
    for key in sorted(set(envs) | set(ENVIRONMENTS)):
        if key not in envs:
            report(FAIL, f"① {key}：index 缺该环境（声明表有）")
            continue
        if key not in ENVIRONMENTS:
            report(FAIL, f"① {key}：index 多出声明表之外的环境")
            continue
        decl = ENVIRONMENTS[key]
        baseline_path = ROOT / decl["baseline"]
        with baseline_path.open(encoding="utf-8") as fh:
            base_image = (yaml.safe_load(fh) or {}).get("image") or {}
        entry = envs[key]
        entry_image = entry.get("image") or {}
        for field in ("ref", "id", "digest"):
            base_val = base_image.get(field)
            idx_val = entry_image.get(field)
            if base_val is None and idx_val is None:
                continue
            if base_val != idx_val:
                report(FAIL, f"① {key} image.{field}：index={idx_val!r} ≠ baseline={base_val!r}")
            else:
                report(PASS, f"① {key} image.{field} 与 baseline 逐字一致")
        if entry.get("env_form") != decl["env_form"]:
            report(FAIL, f"① {key} env_form：index={entry.get('env_form')!r} ≠ 声明表={decl['env_form']!r}")
        if entry.get("stack_lock") != decl["stack_lock"]:
            report(FAIL, f"① {key} stack_lock：index={entry.get('stack_lock')!r} ≠ 声明表={decl['stack_lock']!r}")
        idx_env_dir = (entry.get("env_current") or {}).get("dir")
        if idx_env_dir != decl["env_dir"]:
            report(FAIL, f"① {key} env_current.dir：index={idx_env_dir!r} ≠ 声明表={decl['env_dir']!r}")

    # ── ② id/digest 在 USAGE_MATRIX.md 文本中出现（前 8 位抽取比对） ──
    for key, entry in envs.items():
        entry_image = entry.get("image") or {}
        has_digest = "digest" in entry_image
        for field in ("id", "digest"):
            value = entry_image.get(field)
            if not value:
                continue
            if anchor_prefix(value) in matrix_text:
                report(PASS, f"② {key} {field} 前 8 位 {anchor_prefix(value)} 在 USAGE_MATRIX 出现")
            elif field == "id" and has_digest:
                report(INFO, f"② {key} id 未在 USAGE_MATRIX 展示（该环境展示口径=digest，digest 已核对）")
            else:
                report(FAIL, f"② {key} {field} 前 8 位 {anchor_prefix(value)} 未出现在 USAGE_MATRIX")

    # ── ③ 引用的文件/目录存在 ──
    for key, entry in envs.items():
        entry_image = entry.get("image") or {}
        targets = [("stack_lock", entry.get("stack_lock")), ("image.baseline", entry_image.get("baseline"))]
        env_current = entry.get("env_current") or {}
        targets.append(("env_current.dir", env_current.get("dir")))
        targets.append(("env_current.baseline", env_current.get("baseline")))
        for name, rel in targets:
            if rel is None:
                continue
            path = ROOT / rel
            if name == "env_current.dir":
                ok = path.is_dir()
            else:
                ok = path.is_file()
            if ok:
                report(PASS, f"③ {key} {name} 存在：{rel}")
            else:
                report(FAIL, f"③ {key} {name} 不存在：{rel}")

    # ── ④ 生成物自证 ──
    generated_by = index.get("generated_by") or ""
    if "release_index.py" in generated_by:
        report(PASS, f"④ generated_by 含 release_index.py：{generated_by}")
    else:
        report(FAIL, f"④ generated_by 不含 release_index.py：{generated_by!r}")
    if index.get("schema") == SCHEMA:
        report(PASS, f"④ schema 正确：{SCHEMA}")
    else:
        report(FAIL, f"④ schema 应为 {SCHEMA}，实为 {index.get('schema')!r}")


def main() -> int:
    if not INDEX_PATH.is_file():
        print(f"check_release: 缺 {INDEX_PATH.relative_to(ROOT)}（先跑 release_index.py export）",
              file=sys.stderr)
        return 2
    if not MATRIX_PATH.is_file():
        print(f"check_release: 缺 {MATRIX_PATH.relative_to(ROOT)}", file=sys.stderr)
        return 2
    with INDEX_PATH.open(encoding="utf-8") as fh:
        index = yaml.safe_load(fh) or {}
    matrix_text = MATRIX_PATH.read_text(encoding="utf-8")
    check(index, matrix_text)
    total = fails
    print("-" * 60)
    if fails:
        print(f"check_release: 红（{fails} 项 FAIL / INFO {infos} 项）")
        return 1
    print(f"check_release: 绿（四查全过 / INFO {infos} 项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
