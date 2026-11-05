#!/usr/bin/env python3
"""`DC_MODEL` 口径的离线守卫：**两腿必须共用同一份解析实现**（无设备、无 torch）。

════ 它守的是什么缺陷（2026-10-27）════
  同一个 `DC_MODEL` 环境变量，两条腿理解不一致：
    `proto_infer_leg.py` 有 `resolve_model()`（HF 缓存根 → snapshots/<hash>）⇒ 推理腿过；
    `proto_train_leg.py` 没有 ⇒ 训练腿在同一台机器、同一份
    `DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B` 上直接
    `ValueError: Unrecognized model in …`。
  这类「声明即承诺但只在一半实现里兑现」的问题**不需要真机就能查**——
  与 `backend_offline_check.py` 同一思路：把形态层面的错在本地拦掉。

  处置不是复制一份 `resolve_model` 到训练腿（两份实现必然漂移），
  而是抽成 `runtime/proto/model_path.py` 共用。

════ 判据（3 组，全部离线）════
  G1 解析行为：普通目录 / snapshot 目录 / hub 缓存根（有 refs/main、无 refs/main、
     refs 指坏、空 snapshots、路径不存在）各态返回值与告警是否符合约定。
  G2 **同源守卫**（核心）：AST 静态检查两条腿
       · 不得各自定义 `resolve_model` / `pick_model` / `require_model`（防复制粘贴漂移）
       · 必须从 `model_path` 导入，且必须调用 `require_model(`
       · 不得再出现 `os.environ.get("DC_MODEL"`（防绕开共用实现读环境变量）
  G3 早失败：取不到模型时 `require_model` 必须 `SystemExit` 且**码为 4**，
     且指引文案里三种合法传法都在（否则「早失败」只是把问题提前，没说清怎么给）。

⛔ 它不能证明什么：真机上模型能加载、两腿解析结果在**同一容器视角**下一致。
   后者必须上机实测（四实例复验：两腿各用同一份 DC_MODEL 跑一轮，比对
   `[model] … → 解析 <path>` 那行的路径是否逐字相同）。

用法：
    python3 prototype/scripts/model_path_parity_check.py
    python3 prototype/scripts/model_path_parity_check.py --proto <prototype 目录>
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tempfile
import traceback

PASS = FAIL = SKIPPED = 0

#: 两腿必须**都不定义**的名字（只能来自共用模块）
_FORBIDDEN_DEFS = ("resolve_model", "pick_model", "require_model")
LEGS = ("runtime/proto/proto_infer_leg.py", "runtime/proto/proto_train_leg.py")
SHARED = "runtime/proto/model_path.py"


def ok(name: str, detail: str = "") -> None:
    global PASS
    PASS += 1
    print(f"  [PASS] {name}" + (f" —— {detail}" if detail else ""))


def bad(name: str, detail: str = "") -> None:
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name}" + (f" —— {detail}" if detail else ""))


def skip(name: str, why: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  [SKIP] {name} —— {why}")


# ── G1 解析行为 ──────────────────────────────────────────────────────────────
def g1_behavior(mp) -> None:
    print("\n[G1] 解析行为（合成布局，不需要真模型）")
    with tempfile.TemporaryDirectory() as td:
        # 形态 3：普通目录（自身就是模型目录）
        plain = os.path.join(td, "plain")
        os.makedirs(plain)
        got = mp.resolve_model(plain)
        (ok if got == plain else bad)("普通目录原样使用", got)

        # 形态 1：snapshot 目录本身（父目录有 snapshots/ 但自己不是缓存根）
        hub = os.path.join(td, "models--o--m")
        snaps = os.path.join(hub, "snapshots")
        s1, s2 = os.path.join(snaps, "aaaa1111"), os.path.join(snaps, "bbbb2222")
        os.makedirs(s1)
        os.makedirs(s2)
        got = mp.resolve_model(s1)
        (ok if got == s1 else bad)("snapshot 目录原样使用（不被二次下钻）", got)

        # 形态 2a：缓存根 + refs/main 指向较早那个 ⇒ 必须听 refs/main，**不是**排序最后
        os.makedirs(os.path.join(hub, "refs"))
        with open(os.path.join(hub, "refs", "main"), "w", encoding="utf-8") as fh:
            fh.write("aaaa1111")
        got = mp.resolve_model(hub)
        (ok if got == s1 else bad)("缓存根 + refs/main 优先（refs 旧于排序）",
                                   f"{os.path.basename(got)}（期望 aaaa1111）")

        # 形态 2b：refs/main 指向不存在的 hash ⇒ 回退排序最后
        with open(os.path.join(hub, "refs", "main"), "w", encoding="utf-8") as fh:
            fh.write("deadbeef")
        got = mp.resolve_model(hub)
        (ok if got == s2 else bad)("refs/main 失效 ⇒ 回退排序最后", os.path.basename(got))

        # 形态 2c：无 refs + 多 snapshot ⇒ 取排序最后（并应有告警）
        os.remove(os.path.join(hub, "refs", "main"))
        got = mp.resolve_model(hub)
        (ok if got == s2 else bad)("无 refs/main ⇒ 取排序最后", os.path.basename(got))

        # 空 snapshots/ 与不存在的路径：原样返回，交给 from_pretrained 报错（不猜）
        empty = os.path.join(td, "models--e--m")
        os.makedirs(os.path.join(empty, "snapshots"))
        got = mp.resolve_model(empty)
        (ok if got == empty else bad)("空 snapshots/ 原样返回", got)
        gone = os.path.join(td, "nope")
        (ok if mp.resolve_model(gone) == gone else bad)("路径不存在原样返回", gone)
        (ok if mp.resolve_model("") == "" else bad)("空字符串原样返回", "")

        # ⛔ 不支持「所有模型的父目录」：必须原样返回（不去猜哪个模型）
        hubroot = os.path.join(td, "hub")
        os.makedirs(hubroot)
        os.makedirs(os.path.join(hubroot, "models--o--m2"))
        got = mp.resolve_model(hubroot)
        (ok if got == hubroot else bad)("hub 父目录不猜模型（原样返回）", got)


# ── G3 早失败 ────────────────────────────────────────────────────────────────
def g3_early_fail(mp) -> None:
    print("\n[G3] 早失败：取不到模型必须 exit 4 且说清三种传法")
    os.environ.pop("DC_MODEL", None)
    saved_layouts = mp.INSTANCE_LAYOUTS
    try:
        mp.INSTANCE_LAYOUTS = ("/nonexistent/for/parity/check",)   # 让探测必然落空
        import contextlib
        import io
        buf = io.StringIO()
        code = None
        with contextlib.redirect_stdout(buf):
            try:
                mp.require_model("", "**未找到**（须显式给 DC_MODEL）")
            except SystemExit as exc:
                code = exc.code
        out = buf.getvalue()
        (ok if code == 4 else bad)("取不到模型 ⇒ SystemExit(4)", f"code={code}")
        for form, label in (("普通目录", "普通目录"), ("snapshot", "snapshot"),
                            ("models--", "HF 缓存根")):
            (ok if form in out else bad)(f"指引含「{label}」", "")
        (ok if "/hf_cache/hub" in out else bad)("指引显式劝阻「所有模型的父目录」", "")
    finally:
        mp.INSTANCE_LAYOUTS = saved_layouts


# ── G2 同源守卫 ──────────────────────────────────────────────────────────────
def _parse(path: str) -> ast.Module:
    with open(path, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=path)


def _reads_dc_model_env(tree: ast.Module) -> bool:
    """AST 判「真的读了 `DC_MODEL` 环境变量」（注释/文档串里提到不算 —— 文本匹配会误报）。"""
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "get" and n.args \
                and isinstance(n.args[0], ast.Constant) and n.args[0].value == "DC_MODEL" \
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == "environ":
            return True
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and n.slice.value == "DC_MODEL" \
                and isinstance(n.value, ast.Attribute) and n.value.attr == "environ":
            return True
    return False


def g2_same_source(proto: str) -> None:
    print("\n[G2] 同源守卫（AST 静态检查：两腿不得各自实现解析）")
    shared = os.path.join(proto, SHARED)
    (ok if os.path.isfile(shared) else bad)("共用模块存在", SHARED)
    if not os.path.isfile(shared):
        return

    for rel in LEGS:
        path = os.path.join(proto, rel)
        if not os.path.isfile(path):
            bad(f"{rel} 存在", "文件缺失")
            continue
        tree = _parse(path)
        defs = {n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        dup = sorted(defs & set(_FORBIDDEN_DEFS))
        (ok if not dup else bad)(f"{rel} 未自建解析函数", "、".join(dup) or "无")

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "model_path":
                imported |= {a.name for a in node.names}
        (ok if "require_model" in imported else bad)(
            f"{rel} 从 model_path 导入 require_model", "、".join(sorted(imported)) or "无")

        # 绕开共用实现的写法：直接读环境变量（AST 判，注释里提到不算）
        direct = _reads_dc_model_env(tree)
        (ok if not direct else bad)(f"{rel} 不直接读 DC_MODEL 环境变量", "命中" if direct else "无")

        called = any(isinstance(n, ast.Call) and (
            (isinstance(n.func, ast.Name) and n.func.id == "require_model")
            or (isinstance(n.func, ast.Attribute) and n.func.attr == "require_model"))
            for n in ast.walk(tree))
        (ok if called else bad)(f"{rel} 确实调用了 require_model", "已调用" if called else "未调用")


def main() -> int:
    global FAIL
    ap = argparse.ArgumentParser()
    ap.add_argument("--proto", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    help="prototype 目录（默认＝本脚本上一级）")
    args = ap.parse_args()
    proto = os.path.abspath(args.proto)

    print(f"[model-path-parity] prototype = {proto}")
    sys.path.insert(0, os.path.join(proto, "runtime", "proto"))
    try:
        import model_path as mp
    except Exception:
        print("[FATAL] 无法导入 runtime/proto/model_path.py：")
        print(traceback.format_exc())
        return 1

    try:
        g1_behavior(mp)
        g2_same_source(proto)
        g3_early_fail(mp)
    except Exception:
        FAIL += 1
        print("\n[FATAL] 自检本身抛出未预期异常（已判 1 条失败）")
        print(traceback.format_exc())

    print("\n" + "=" * 74)
    print(f"DC_MODEL 口径自检: {PASS} 通过 / {FAIL} 失败 / {SKIPPED} 跳过")
    print("⚠️ 结论边界：本结果只证明**解析形态与同源关系**。"
          "真机上两腿解析到同一路径必须上机实测（比对 `[model] … → 解析 <path>` 行）。")
    print("=" * 74)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
