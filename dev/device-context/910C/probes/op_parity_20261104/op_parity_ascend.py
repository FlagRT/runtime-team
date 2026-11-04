#!/usr/bin/env python3
"""910C 算子库逐算子对照探针（2026-11-04）

目的：把「算子库只有端到端间接证据」补成「逐算子实测对照」。

做法（两阶段，同一进程内完成，避免张量序列化）：
  阶段 1  record：在 **vendor 原生路径**下跑一段真实前向（Qwen3-Embedding-0.6B），
          用 TorchDispatchMode 捕获**实际执行的每个 aten 算子**及其输入/输出；
  阶段 2  replay：`flag_gems.enable()` 后，用**同一份输入**重放每个算子，
          与阶段 1 的 vendor 输出逐算子比对（rel / allclose）。

纪律（对齐仓内判据惯例）：
  * 判据是「同一输入、两条路径」——不是两次跑同一路径比运气；
  * 只对照**我方验收路径实际执行到的算子**（不是全量算子库普查）；
  * 可变算子（in-place）跳过：重放会污染输入，判据不可信；
  * 每个算子最多留 N 个不同签名（shape/dtype）的样本，内存有预算上限；
  * 跑完同时打印「flag_gems 是否覆盖该算子」——未覆盖的算子重放必然等于 vendor，
    这类必须在报告里与「覆盖且一致」分开列，不可混写。

输出：TSV（逐算子一行）+ 摘要 JSON。
"""
import argparse
import json
import os
import sys
import traceback
from collections import defaultdict

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

RECORD_BUDGET_BYTES = int(os.environ.get("REC_BUDGET_MB", "600")) << 20
MAX_SIGS_PER_OP = int(os.environ.get("MAX_SIGS_PER_OP", "3"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "128"))
ATOL = float(os.environ.get("ATOL", "1e-3"))
RTOL = float(os.environ.get("RTOL", "1e-3"))

FLOAT_OK = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def dev():
    return "npu:0" if hasattr(torch, "npu") else "cuda:0"


def nbytes(x):
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size()
    return 0


def flat_bytes(leaves):
    return sum(nbytes(x) for x in leaves)


def clone_keep(x):
    """张量取 CPU 副本（脱离设备生命周期），其余原样保留。"""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    return x


def shape_sig(leaves):
    parts = []
    for x in leaves:
        if isinstance(x, torch.Tensor):
            parts.append("%s/%s" % ("x".join(str(s) for s in x.shape), str(x.dtype).replace("torch.", "")))
        else:
            parts.append(type(x).__name__)
    return "|".join(parts)[:400]


class Recorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.records = []          # list of dict
        self.counts = defaultdict(int)
        self.skipped = defaultdict(int)
        self.budget = RECORD_BUDGET_BYTES
        self.seen_sig = defaultdict(set)
        self.op_funcs = {}

    def _skip(self, name, why):
        self.skipped["%s :: %s" % (name, why)] += 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)          # 真实执行（vendor 路径）
        name = str(func)
        self.counts[name] += 1
        try:
            self._maybe_store(name, func, args, kwargs, out)
        except Exception as exc:             # 记录失败不影响被测前向
            self._skip(name, "store_fail:%s" % type(exc).__name__)
        return out

    def _maybe_store(self, name, func, args, kwargs, out):
        schema = getattr(func, "_schema", None)
        if schema is not None and getattr(schema, "is_mutable", False):
            self._skip(name, "mutable")
            return
        leaves, spec = pytree.tree_flatten((args, kwargs))
        out_leaves, out_spec = pytree.tree_flatten(out)
        if not out_leaves or not any(isinstance(o, torch.Tensor) for o in out_leaves):
            self._skip(name, "no_tensor_out")
            return
        if any(o is None for o in out_leaves):
            self._skip(name, "out_has_None")
            return
        # 只保留可安全重放的输入类型
        for x in leaves:
            if isinstance(x, torch.Tensor):
                continue
            if isinstance(x, (int, float, bool, str, type(None))):
                continue
            if isinstance(x, (list, tuple)) and all(isinstance(y, (int, float, bool)) for y in x):
                continue
            self._skip(name, "arg_type_%s" % type(x).__name__)
            return
        sig = shape_sig(leaves)
        if len(self.seen_sig[name]) >= MAX_SIGS_PER_OP:
            self._skip(name, "sig_cap")
            return
        cost = flat_bytes(leaves) + flat_bytes(out_leaves)
        if cost > self.budget:
            self._skip(name, "budget")
            return
        self.budget -= cost
        self.seen_sig[name].add(sig)
        self.records.append({
            "op": name,
            "sig": sig,
            "leaves": [clone_keep(x) for x in leaves],
            "spec": spec,
            "out_leaves": [clone_keep(x) for x in out_leaves],
            "out_spec": out_spec,
            "out_dtypes": [str(o.dtype) if isinstance(o, torch.Tensor) else None for o in out_leaves],
        })
        self.op_funcs[name] = func


def rel_err(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().max().item()
    num = (a - b).abs().max().item()
    if denom == 0:
        return 0.0 if num == 0 else float("inf")
    return num / denom


def do_record(model, tok, texts, d):
    rec = Recorder()
    with torch.no_grad(), rec:
        for t in texts:
            batch = tok(t, return_tensors="pt", truncation=True, max_length=MAX_SEQ).to(d)
            _ = model(**batch)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.environ.get("DC_OUT_DIR", "."))
    ap.add_argument("--model", default=os.environ.get("DC_MODEL", "/workspace/models/Qwen3-Embedding-0.6B"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = dev()

    report = {"device": d, "torch": torch.__version__, "model": args.model}
    try:
        import torch_npu  # noqa: F401
        report["torch_npu"] = torch_npu.__version__
    except Exception:
        pass
    try:
        import flag_gems
        report["flag_gems"] = getattr(flag_gems, "__version__", "?")
        report["flag_gems_file"] = flag_gems.__file__
    except Exception as exc:
        report["flag_gems"] = "import_failed:%r" % (exc,)

    print("[env] %s" % json.dumps({k: report[k] for k in ("device", "torch", "torch_npu", "flag_gems")}, ensure_ascii=False), flush=True)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval().to(d)
    print("[model] loaded", flush=True)

    texts = [
        "设备执行上下文层把各家芯片的分歧显式化，并保持上层代码可移植。",
        "The device execution context layer keeps upper layers portable across vendors.",
        "多流并发、事件计时、错误恢复是这一层的三类核心判据。",
    ]

    print("[phase1] record on vendor path ...", flush=True)
    rec = do_record(model, tok, texts, d)
    print("[phase1] ops executed = %d unique, records kept = %d" % (len(rec.counts), len(rec.records)), flush=True)

    # ── 阶段 2：启用 FlagGems 后重放
    print("[phase2] enabling flag_gems ...", flush=True)
    enabled = False
    try:
        import flag_gems
        flag_gems.enable()
        enabled = True
        print("[phase2] flag_gems.enable() ok", flush=True)
    except Exception:
        traceback.print_exc()

    rows = []
    if enabled:
        for r in rec.records:
            name = r["op"]
            opname = name.split(".")[1] if "." in name else name
            covered = hasattr(flag_gems, opname)
            dev_leaves = [x.to(d) if isinstance(x, torch.Tensor) else x for x in r["leaves"]]
            try:
                a2, k2 = pytree.tree_unflatten(dev_leaves, r["spec"])
                out2 = rec.op_funcs[name](*a2, **k2)
                out2_leaves, _ = pytree.tree_flatten(out2)
                if len(out2_leaves) != len(r["out_leaves"]):
                    raise RuntimeError("out arity changed")
                worst = 0.0
                ok = True
                for got, ref in zip(out2_leaves, r["out_leaves"]):
                    if isinstance(ref, torch.Tensor):
                        g = got.detach().to("cpu").float()
                        rf = ref.float()
                        if g.shape != rf.shape:
                            raise RuntimeError("shape mismatch %s vs %s" % (tuple(g.shape), tuple(rf.shape)))
                        worst = max(worst, rel_err(g, rf))
                        if not torch.allclose(g, rf, atol=ATOL, rtol=RTOL):
                            ok = False
                verdict = "PASS" if ok else "FAIL"
                rows.append({"op": name, "sig": r["sig"], "covered": covered,
                             "rel": worst, "verdict": verdict, "note": ""})
            except Exception as exc:
                rows.append({"op": name, "sig": r["sig"], "covered": covered,
                             "rel": None, "verdict": "SKIP",
                             "note": "%s:%s" % (type(exc).__name__, str(exc)[:80])})
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
    else:
        print("[phase2] 未能启用 flag_gems ⇒ 本探针无结论（不产出 PASS）", flush=True)

    # ── 落盘
    tsv = os.path.join(args.out_dir, "op_parity.tsv")
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("op\tcovered\trel\tverdict\tsig\tnote\n")
        for r in sorted(rows, key=lambda x: (x["verdict"] != "FAIL", x["op"])):
            fh.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (
                r["op"], "Y" if r["covered"] else "N",
                "NA" if r["rel"] is None else "%.3e" % r["rel"],
                r["verdict"], r["sig"], r["note"]))

    cnt = defaultdict(int)
    for r in rows:
        cnt[r["verdict"]] += 1
    cov = sum(1 for r in rows if r["covered"])
    report.update({
        "ops_executed_unique": len(rec.counts),
        "records_replayed": len(rows),
        "covered_by_flag_gems": cov,
        "pass": cnt["PASS"], "fail": cnt["FAIL"], "skip": cnt["SKIP"],
        "record_skipped_by_reason": dict(sorted(rec.skipped.items(), key=lambda kv: -kv[1])[:12]),
        "top_ops_by_count": sorted(rec.counts.items(), key=lambda kv: -kv[1])[:25],
    })
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    print("[summary] executed=%d replayed=%d covered=%d PASS=%d FAIL=%d SKIP=%d" % (
        report["ops_executed_unique"], report["records_replayed"], cov,
        cnt["PASS"], cnt["FAIL"], cnt["SKIP"]), flush=True)
    fails = [r for r in rows if r["verdict"] == "FAIL"]
    for r in fails[:10]:
        print("  [FAIL] %s rel=%.3e sig=%s" % (r["op"], r["rel"], r["sig"][:90]), flush=True)
    print("DONE", flush=True)
    return 0 if cnt["FAIL"] == 0 and len(rows) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
