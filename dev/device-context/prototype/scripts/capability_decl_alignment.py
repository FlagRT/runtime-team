#!/usr/bin/env python3
"""四家后端「能力声明」对齐核验（芯片无关）。

提取每个后端的 `_capabilities`（声明集）与 `_CAPABILITY_KEYS`（全集），
剥掉注释行后比对：
  * 声明集 ⊆ 全集？（声明了全集里没有的键 ⇒ 对称性缺口）
  * 四家的全集是否一致？（不一致 ⇒ 跨后端读到的键空间不同）
  * 逐键：谁声明了、谁没声明（这是**真实差异**，需在报告里如实解释）
"""
import pathlib
import re
import sys

BACKENDS = ["ascend", "kunlun", "cambricon", "ppu"]
ROOT = pathlib.Path(__file__).resolve().parents[1] / "runtime" / "backends"


def _strip_comments(src: str) -> str:
    """剥掉整行注释与**行尾注释**（能力清单里只有字符串与逗号，不会含 `#`）。"""
    out = []
    for line in src.splitlines():
        s = line.lstrip()
        if s.startswith("#"):
            continue
        i = line.find("#")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def _extract(src: str, attr: str):
    """按**赋值语句**精确提取 `attr = {` / `attr = (` 的内容（避免匹配到注释或别的标识符）。"""
    src = _strip_comments(src)
    m = re.search(rf"\b{re.escape(attr)}\s*=\s*([\(\{{])", src)
    if not m:
        return None
    start = m.start(1)
    close = "}" if src[start] == "{" else ")"
    depth = 0
    for p in range(start, len(src)):
        if src[p] == src[start]:
            depth += 1
        elif src[p] == close:
            depth -= 1
            if depth == 0:
                return src[start + 1:p]
    return None


def keys_of(body: str):
    out, seen = [], set()
    for m in re.finditer(r'["\']([A-Za-z_][A-Za-z_0-9]*)["\']', _strip_comments(body)):
        k = m.group(1)
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def main():
    decl, full = {}, {}
    for b in BACKENDS:
        p = ROOT / b / "backend.py"
        src = p.read_text(encoding="utf-8")
        d = _extract(src, "_capabilities")
        f = _extract(src, "_CAPABILITY_KEYS")
        decl[b] = keys_of(d) if d else []
        full[b] = keys_of(f) if f else []
        if not d:
            print(f"⚠️ {b}: 未提取到 _capabilities")
        if not f:
            print(f"⚠️ {b}: 未提取到 _CAPABILITY_KEYS")

    print("=" * 92)
    print("① 声明集 ⊆ 全集？")
    print("=" * 92)
    bad = 0
    for b in BACKENDS:
        extra = sorted(set(decl[b]) - set(full[b]))
        print(f"  {b:10s} 声明 {len(decl[b]):2d} / 全集 {len(full[b]):2d}   "
              f"声明集越界: {extra or '无 ✅'}")
        bad += len(extra)

    print()
    print("=" * 92)
    print("② 四家「全集」是否一致")
    print("=" * 92)
    allk = sorted(set().union(*[set(full[b]) for b in BACKENDS]))
    for b in BACKENDS:
        miss = sorted(set(allk) - set(full[b]))
        print(f"  {b:10s} 缺: {miss or '无 ✅'}")

    print()
    print("=" * 92)
    print("③ 逐键声明矩阵（✅ 声明 / · 未声明）")
    print("=" * 92)
    print(f"  {'能力键':<26}" + "".join(f"{b:>12}" for b in BACKENDS))
    for k in allk:
        row = f"  {k:<26}"
        for b in BACKENDS:
            row += f"{('✅' if k in decl[b] else '·'):>12}"
        print(row)

    print()
    print("=" * 92)
    print("④ 差异汇总（只有部分家声明的键 —— 每条都要在报告里有如实解释）")
    print("=" * 92)
    for k in allk:
        who = [b for b in BACKENDS if k in decl[k and k or b]] if False else [b for b in BACKENDS if k in decl[b]]
        if 0 < len(who) < len(BACKENDS):
            print(f"  {k:<26} 声明者: {', '.join(who)}   （缺: {', '.join(b for b in BACKENDS if b not in who)}）")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
