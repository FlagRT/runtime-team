#!/usr/bin/env python3
"""根解析自检 v2（AST 版）：仓库内脚本自算「原型根」是否算对（2026-09-30 新增）。

**为什么需要**：同一天因为「根解析 off-by-one」踩了两次 ——
  ① `runtime/demos/demo_unified.py` 写 `parents[1]`（应为 `parents[2]`）：靠"调用方 CWD 恰好在根"
     侥幸可用（910C 容器 `PYTHONPATH` 末尾那个 `:` 把 CWD 带进 `sys.path`），P800 直接
     `ModuleNotFoundError: No module named 'runtime'`；
  ② 新增探针写 `dirname(dirname(HERE))`（应为 `dirname(HERE)`）⇒ 真机 `ModuleNotFoundError`。
这类错**只在真机上暴露**，但完全可以**离线静态查出来**。

判据（写死，不事后调整；用 **AST** 而非正则 —— v1 用正则**被行尾注释挡住**，恰好漏掉 ②）：

  **A（强判据）**：每个 `sys.path.insert(...)` / `sys.path.append(...)` 的**路径实参**，
      在该文件的真实路径下求值后，必须（规范化）**等于原型根** `prototype/`
      —— 因为它的目的就是让 `import runtime` 可用。
  **B**：每个**名字含 `ROOT`** 且**由 `__file__` 派生**的赋值，求值后必须等于原型根。
  **C（信息性）**：没有任何自解析、只靠 `DC_ROOT` 的脚本会被列出（真机用 `-e DC_ROOT=` 传入 ⇒ 不算失败）。

⚠️ 安全：只 eval「白名单字符 + 必须引用文件路径派生物」的表达式，且只读仓库自己的文件。
"""
import argparse
import ast
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                        # prototype/

SCAN_DIRS = ["probes", "runtime", "scripts"]
SAFE_EXPR = re.compile(r"^[A-Za-z0-9_\.\(\)\[\]\"'/, :]+$")


def target_files():
    out = []
    for d in SCAN_DIRS:
        for dirpath, _dirs, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in dirpath:
                continue
            out += [os.path.join(dirpath, fn) for fn in files if fn.endswith(".py")]
    return sorted(out)


def _is_file_based(expr: str, known: dict) -> bool:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    return ("__file__" in toks) or any(t in toks for t in known if "HERE" in t or "DIR" in t)


def _eval(expr: str, path: str, names: dict):
    """在「该文件真实路径 + 已求值变量」下求值（白名单 + 只读自家文件）。"""
    flat = re.sub(r"\s+", " ", expr).strip()
    if not SAFE_EXPR.match(flat):
        return None, f"含白名单外字符：{flat[:70]!r}"
    ns = {"os": os, "sys": sys, "__file__": path}
    from pathlib import Path
    ns["Path"] = Path
    ns.update(names)
    try:
        # ⚠️ 返回**原始值**（Path 等），不要在这里就 `str()`/`abspath()` ——
        #    否则后续表达式如 `str(_HERE.parent)` 会对**字符串**取 `.parent` 而失败
        #    （2026-09-30 实测踩到：`AttributeError: 'str' object has no attribute 'parent'`）。
        return eval(flat, ns), None                                  # noqa: S307
    except Exception as e:                                           # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:70]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet-ok", action="store_true")
    a = ap.parse_args()
    print(f"[env] 原型根 = {ROOT}\n[env] 扫描 = {SCAN_DIRS}\n")

    n_ok = n_bad = 0
    bad, dcroot_only, unparsed = [], [], []

    for f in target_files():
        rel = os.path.relpath(f, ROOT)
        try:
            src = io.open(f, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception as e:                                       # noqa: BLE001
            print(f"  [SKIP] {rel}: 解析失败 {e}")
            continue

        names: dict = {}                 # 文件内已求值的变量（按出现顺序）
        assigns = []                     # (name, expr, is_file_based)
        inserts = []                     # (实参源码, 行号)
        imports_runtime = False          # 该文件是否 import runtime（判据 A1 的前提）

        for node in tree.body:
            # 赋值（顶层）
            if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                    and isinstance(node.targets[0], ast.Name):
                nm = node.targets[0].id
                expr = ast.unparse(node.value)
                val, err = _eval(expr, f, names)
                if val is not None:
                    names[nm] = val                # 存**原始值**（后续表达式还要用它）
                    assigns.append((nm, expr, _is_file_based(expr, names)))
                elif nm.upper().endswith("ROOT") or "HERE" in nm:
                    unparsed.append((rel, nm, expr, err))
            # 是否 import runtime（判据 A1 的前提）
            if isinstance(node, ast.Import):
                if any(al.name.split(".")[0] == "runtime" for al in node.names):
                    imports_runtime = True
            if isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] == "runtime":
                    imports_runtime = True
            # sys.path.insert/append
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                fn = ast.unparse(call.func)
                if fn in ("sys.path.insert", "sys.path.append") and len(call.args) >= 1:
                    arg = call.args[-1]
                    inserts.append((ast.unparse(arg), getattr(node, "lineno", 0)))

        # ── 判据 A1（强）：import runtime 的脚本必须把**原型根**插进 sys.path ──
        #    为什么不是「必须等于根」：部分脚本**刻意**把 `runtime/proto`、`runtime/conformance`
        #    也插进来（扁平名导入，见台账第 23 条）—— 那是合法用法，判「必须等于根」会**误报**。
        vals = []
        for expr, lineno in inserts:
            val, err = _eval(expr, f, names)
            if val is not None:
                val = os.path.abspath(str(val))        # 比较用规范路径字符串
            if val is None:
                print(f"  [INFO] {rel}:{lineno} sys.path.insert 实参不可静态求值（{err}）")
                continue
            vals.append((lineno, expr, val))
        if imports_runtime and inserts:
            has_root = any(os.path.normpath(v) == os.path.normpath(ROOT) for _l, _e, v in vals)
            if has_root:
                n_ok += 1
                if not a.quiet_ok:
                    print(f"  [PASS] {rel}: import runtime 且 sys.path 含原型根")
            elif "DC_ROOT" in src:
                # 例外（项目规定做法）：脚本从 `DC_ROOT` 环境变量取根 ⇒ 真机由 `-e DC_ROOT=`
                # 显式传入（本方向所有真机命令都这么传）⇒ **不算失败**，只记信息。
                print(f"  [INFO] {rel}: 根来自 `DC_ROOT`（无自解析）—— 真机须显式传入，属规定做法")
            else:
                n_bad += 1
                bad.append((rel, "import runtime 但 sys.path 无原型根", " | ".join(e for _l, e, _v in vals), ""))
                print(f"  [FAIL] {rel}: **import 了 runtime 但 sys.path 里没有原型根** "
                      f"⇒ 换 CWD 就会 ModuleNotFoundError")
                for _l, e, v in vals:
                    print(f"         :{_l} → {v}   （{e[:70]}）")

        # ── 判据 A2：每个 sys.path 实参必须**存在**且落在 `prototype/` 之内 ──
        for lineno, expr, val in vals:
            inside = (os.path.normpath(val) == os.path.normpath(ROOT)
                      or os.path.normpath(val).startswith(os.path.normpath(ROOT) + os.sep))
            exists = os.path.isdir(val)
            if inside and exists:
                n_ok += 1
                if not a.quiet_ok:
                    print(f"  [PASS] {rel}:{lineno} sys.path 实参在原型根内且存在 → {val}")
            else:
                n_bad += 1
                why = "不在 prototype/ 之内" if not inside else "目录不存在"
                bad.append((rel, f"sys.path.insert@{lineno}", expr, val))
                print(f"  [FAIL] {rel}:{lineno} sys.path 实参{why} → {val}")
                print(f"         表达式：{expr[:100]}")

        # ── 判据 B：名字含 ROOT 的 file-based 赋值必须是原型根 ──
        for nm, expr, file_based in assigns:
            if ("ROOT" not in nm.upper()) or not file_based:
                continue
            val = os.path.abspath(str(names.get(nm)))
            if os.path.normpath(val) == os.path.normpath(ROOT):
                n_ok += 1
                if not a.quiet_ok:
                    print(f"  [PASS] {rel} {nm} → {val}")
            else:
                n_bad += 1
                bad.append((rel, nm, expr, val))
                print(f"  [FAIL] {rel} {nm} → {val}（应为 {ROOT}）")
                print(f"         表达式：{expr[:110]}")

        if not inserts and not any("ROOT" in n.upper() for n, _e, _fb in assigns):
            if "DC_ROOT" in src:
                dcroot_only.append(rel)

    print(f"\n=== 汇总：正确 {n_ok} / 错误 {n_bad} ===")
    if unparsed:
        print(f"  [信息] 无法静态求值的根类赋值（{len(unparsed)} 个，多为默认值含绝对路径）：")
        for r, nm, expr, err in unparsed[:8]:
            print(f"         {r} {nm}: {err}")
    if dcroot_only:
        print(f"  [信息] 仅依赖 DC_ROOT（无自解析，{len(dcroot_only)} 个）：")
        for r in dcroot_only[:12]:
            print(f"         {r}")
    verdict = "ROOT_RESOLUTION_PASS" if n_bad == 0 else "ROOT_RESOLUTION_FAIL"
    print(f"verdict = {verdict}")
    print("⚠️ 边界：静态检查是「真机 import 成功」的**必要不充分**条件 —— "
          "它抓的是路径算错，不替代真机执行。")

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fp:
            json.dump({"root": ROOT, "ok": n_ok, "bad": n_bad, "verdict": verdict,
                       "bad_detail": [{"file": r, "what": w, "expr": e, "got": g}
                                      for r, w, e, g in bad],
                       "dc_root_only": dcroot_only,
                       "unparsed": [{"file": r, "var": v, "err": er} for r, v, _e, er in unparsed]},
                      fp, ensure_ascii=False, indent=2)
        print(f"[out] {a.out}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
