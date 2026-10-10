#!/usr/bin/env python3
"""同口径证据矩阵扫描（芯片无关 · 只认产物文件，不认文档声明）。

用途：对每个芯片实例，按接入手册 §7 验收清单 13 项 + `VERIFICATION_MANIFEST` §1 + 后续扩展步骤，
     在 `<chip>/probes/` 下**递归**查找产物文件（.log/.json/…）与代码事实，产出「步骤 × 芯片」矩阵。
     找不到 => 该项为缺口（**不是**"文档里提过"）。

纪律（每一条都是被误报打出来的）：
  * 只扫 `probes/` 目录，**不扫 `docs/`** —— 文档提及 = 假证据；
  * **必须递归**（产物常在 `m1_*_out/`、`d2_*_out/` 子目录里）；
  * 命中分两列：**文件名命中** / **内容判据命中** —— 只有文件在但内容是空的标 ⚠️；
  * 支持 `content_only=True` 的步骤（如"非空转验证"没有独立产物，只能是内容级事实）；
  * 可只跑部分实例（命令行传实例名）。

用法：
    python3 prototype/scripts/onboarding_evidence_matrix.py            # 四家全跑
    python3 prototype/scripts/onboarding_evidence_matrix.py PPU P800   # 只跑指定实例
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]   # dev/device-context（本脚本在 prototype/scripts/ 下）
CHIPS = ["910C", "P800", "MLU590", "PPU"]

# (sid, 名称, 文件名正则列表, 内容判据正则列表, content_only, scope)
#   scope: "probes"（默认，只扫 <chip>/probes/）| "docs"（扫 <chip>/docs/，**只查"交付物是否存在"**）
#   ⚠️ scope="docs" 只用于「该实例应有这份报告」这类**交付物存在性**检查，
#      **不得**用来把"文档里提过某步骤"当作步骤证据（那就是假证据）。
STEPS = [
    ("S1", "环境普查 preflight",
     [r"preflight_env.*\.log"], [r"PREFLIGHT|环境"], False, "probes"),
    ("S2", "厂商栈判别",
     [r"stack_survey|stack_probe|A3A4_stack|capability_probe|accept_probe_results|probe_ppu_stack"],
     [r"device_count|is_available|npu|mlu|cuda|ppu|devices"], False, "probes"),
    ("S3", "离线契约自检",
     [r"offline.*\.log|backend_offline"], [r"通过\s*/|PASS|FAIL"], False, "probes"),
    ("S4", "conformance 13 例",
     [r"conf13|conformance_13"], [r"13\s*/\s*13|passed|PASS"], False, "probes"),
    ("S5", "conformance 推理 6 例",
     [r"confinfer6|conformance_infer6|infer_6"], [r"6\s*/\s*6|passed|PASS"], False, "probes"),
    ("S6", "smoke 接入自检",
     [r"smoke"], [r"SMOKE|通过\s*/|失败"], False, "probes"),
    ("S7", "训练腿 50 步",
     [r"train.*leg.*\.(json|log)|E_train|A7_train|m1_legs|train_leg"],
     [r"TRAIN_LEG_(PASS|FAIL)|\"verdict\"|steps"], False, "probes"),
    ("S8", "推理腿（前向）",
     [r"infer_leg.*\.(json|log)|F_infer_leg|accept_inferleg"],
     [r"INFER_LEG_(PASS|FAIL)|\"verdict\"|passed"], False, "probes"),
    ("S9", "推理腿（服务化）",
     [r"serve|vllm"], [r"SERVE_STANDARD_PASS|SERVE_LEG_PASS|ready"], False, "probes"),
    ("S10", "错误闭环",
     [r"error.*loop|errorloop|error_recovery"], [r"闭环|verdict|PASS|OK"], False, "probes"),
    ("S11", "多流 Stream 语义探针（8 项）",
     [r"stream_semantics"], [r"STREAM_SEMANTICS_PASS"], False, "probes"),
    ("S12", "图捕获 V2",
     [r"graph_capture"], [r"GRAPH_CAPTURE_PASS"], False, "probes"),
    ("S13", "流配额",
     [r"stream_quota|available_num"], [r"STREAM_QUOTA_PASS"], False, "probes"),
    ("S14", "B/C 契约探针",
     # ⚠️ MLU590 的产物叫 `m1_bc_probe_cambricon.json` —— 首版只认 bc_contract 名 ⇒ 误报「从未跑」
     [r"bc_contract|bc_probe"], [r"PASS"], False, "probes"),
    ("S15", "职责响应审计（78 项口径）",
     [r"duty_audit|regress_duty|audit_2026|_duty_"],
     [r"DUTY_RESPONSE|\"verdict\"|OK"], False, "probes"),
    ("S16", "A2 多卡多进程恢复压测",
     # ⚠️ 别用裸 `a2_`：`I_base_kl3_A2_KL3on`（KL3 实验标签）会误命中；`sha256` 含 "a2" 同理会误命中
     [r"recover_multiproc"], [r"A2_BOUNDARY|PASS|verdict"], False, "probes"),
    ("S17", "流优先级：区间 / 回读 / 控制",
     # 优先级判据在 stream_semantics 里也有一条（`S12_stream_priority`）⇒ 两类文件都算
     [r"priorit|stream_semantics"], [r"S12_stream_priority|PASS|verdict"], False, "probes"),
    ("S18", "契约不变式 I1–I4",
     [r"coninvariant|contract_invar"], [r"CONTRACT_INVARIANTS_(PASS|FAIL)"], False, "probes"),
    ("S19", "步 0 模型完整性（含损坏取证）",
     [r"MODEL_(CORRUPTION|INTEGRITY)|model_integrity|model_manifest|loadcheck"],
     [r"sha256|PASS|一致|NaN"], False, "probes"),
    ("S20", "回归 / 改前对照复跑（r2+ / PRE_FIX / 前后对照）",
     # ⚠️ 别只认 `regress*`：PPU 用的是 `l1_fix_20261010_out/{pre,post,post5}` 命名
     [r"regress|_r[2-9]\.|PRE_FIX|BLOCKED|_fix_\d{8}_out|/pre_run|/post_run"],
     [r"PASS|FAIL|如实|阻断|blocked"], False, "probes"),
    ("S21", "非空转验证（内容级事实，无独立产物）",
     [], [r"非空转|PRE_FIX|注入后|去掉.{0,12}即 ?FAIL"], True, "probes"),
    ("S22", "⭐ 流优先级「调度效果」对照（芯片无关探针）",
     [r"sched_effect|priority_effect|priority_queued"], [r"NOT_APPLICABLE|PASS|FAIL|verdict"], False, "probes"),
    ("S23", "证据目录 .gitignore `!*.log` 例外",
     [r"\.gitignore"], [r"!\*\.log|\*\.log"], False, "probes"),
    ("D1", "（交付物）多流 16 项基线逐项比对报告",
     [r"STREAM_BASELINE_16|DC_STREAM_MAPPING"], [r"S-1|S-16|16 项"], False, "docs"),
    ("D2", "（交付物）证据索引 `EVIDENCE_INDEX_*`",
     [r"EVIDENCE_INDEX"], [r"probes|证据"], False, "docs"),
]


def scan_chip(chip: str):
    rows = []
    for sid, name, pats, cpat, content_only, scope in STEPS:
        base = ROOT / chip / ("probes" if scope == "probes" else "docs")
        if not base.exists():
            rows.append((sid, name, [], []))
            continue
        files = [p for p in base.rglob("*") if p.is_file()]
        f_hits, c_hits = [], []
        for p in files:
            try:
                txt = "" if p.stat().st_size > 3_000_000 else p.read_text(errors="replace")
            except Exception:
                txt = ""
            rel = f"{p.parent.name}/{p.name}" if p.parent != base else p.name
            hit_cont = any(re.search(c, txt, re.I) for c in cpat)
            if content_only:
                if hit_cont:
                    c_hits.append(rel)
                continue
            if any(re.search(pt, rel, re.I) for pt in pats):
                f_hits.append(rel)
                if hit_cont:
                    c_hits.append(rel)
        rows.append((sid, name, f_hits, c_hits))
    return rows


def main():
    chips = [a for a in sys.argv[1:] if not a.startswith("-")] or CHIPS
    res = {c: scan_chip(c) for c in chips}
    print("=" * 108)
    print("同口径证据矩阵（只认产物文件；`文件命中/判据命中`；❌=无产物，⚠️=有文件无判据串）")
    print("=" * 108)
    print(f"{'步骤':<5}{'名称':<40}" + "".join(f"{c:>15}" for c in chips))
    print("-" * 108)
    for i, (sid, name, _, _c) in enumerate(res[chips[0]]):
        line = f"{sid:<5}{name:<40}"
        for c in chips:
            _, _, f_hits, c_hits = res[c][i]
            if not f_hits and not c_hits:
                cell = "❌"
            elif c_hits:
                cell = f"✅ {len(f_hits)}/{len(c_hits)}" if f_hits else f"✅ 内容{len(c_hits)}"
            else:
                cell = f"⚠️ 文件{len(f_hits)}/判据0"
            line += f"{cell:>15}"
        print(line)
    print()
    print("=" * 108)
    print("逐步骤命中文件（❌ 的项列出，便于人工确认是真缺还是命名不同）")
    print("=" * 108)
    for c in chips:
        print(f"\n########## {c} ##########")
        for sid, name, f_hits, c_hits in res[c]:
            mark = "✅" if c_hits else ("⚠️" if f_hits else "❌")
            print(f"  {mark} {sid} {name}")
            if not f_hits and not c_hits:
                print("        （无产物文件 —— 需人工确认是真缺口还是命名不同）")
                continue
            show = c_hits or f_hits
            for f in show[:5]:
                print(f"        - {f}" + ("" if f in c_hits else "  [文件命中但无判据串]"))
            if len(show) > 5:
                print(f"        …… 另 {len(show)-5} 个")


if __name__ == "__main__":
    main()
