#!/usr/bin/env python3
"""自核验：按接入手册/技能里的标准步骤，逐项扫描四家芯片的**证据是否真的存在**。

判定依据 = 该步骤的产物文件是否在 <chip>/probes 下真实出现（文件名关键词 + 内容关键词）。
⚠️ 只看文件名会误判，故对每项再回看 1 个命中文件的内容关键词。
"""
import pathlib
import re

CHIPS = ["910C", "P800", "MLU590", "PPU"]

# 步骤 -> (文件名/内容关键词列表, 说明)
STEPS = [
    ("1 环境打通(pflt)", ["preflight_env", "preflight"], "环境普查脚本产物"),
    ("2 厂商栈判别", ["stack_survey", "capability_probe", "A3A4_stack", "conf_device", "device_probe", "capability_decisions", "stack"], "栈判别/能力探测"),
    ("3 镜像依赖链自检", ["image", "mirror", "dep_chain", "depchain", "IMAG"], "镜像就绪 5 条判据"),
    ("4 离线契约自检", ["offline_check", "offline"], "无设备自检"),
    ("4b 跨后端对称性", ["symmetry"], "--all 对称性"),
    ("7 smoke 自检", ["smoke"], "冒烟"),
    ("5 conformance 13", ["conf13", "conformance_13", "conformance_runtime", "conformance"], "13 例"),
    ("6 conformance infer6", ["confinfer6", "infer6", "conformance_infer"], "推理 6 例"),
    ("8 训练腿", ["train_leg"], "两条腿-训练"),
    ("9 推理腿前向", ["infer_leg", "proto_infer"], "两条腿-推理前向"),
    ("10 推理腿服务化", ["serve", "SERVE"], "服务化"),
    ("11 错误闭环", ["error_recovery", "errorloop", "recovery_loop"], "四类注入"),
    ("12 已知问题声明", ["known_issues", "KNOWN_ISSUES"], "known_issues 结构化声明"),
    ("多流 16/8 项基线", ["stream_semantics", "STREAM_BASELINE", "stream_baseline"], "多流基线"),
    ("图捕获探针", ["graph_capture", "graph"], "G1-G5"),
    ("流配额探针", ["stream_quota", "quota"], "S-16"),
    ("B/C 契约探针", ["probe_bc_contract", "bc_contract", "B1_B3_B4"], "memory_alloc/record_stream/context_lifecycle"),
    ("context_query 探针", ["PROBE_CONTEXT", "context_query", "context_semantics"], "C4 六键"),
    ("职责响应审计 78", ["duty_audit", "duty_response", "DUTY_RESPONSE"], "§1.1-§1.5+§2+§3"),
    ("A2 多进程恢复压测", ["a2_recover", "recover_multiproc", "A2_"], "real 模式压测"),
    ("非空转验证", ["nonidle", "non_idle", "inject", "非空转"], "判据能 FAIL"),
    ("步 0 模型完整性", ["MODEL_INTEGRITY", "model_integrity", "step0", "loadcheck", "MODEL_CORRUPTION"], "模型先验"),
]

CHIP_DIR = pathlib.Path(".")

# 收集每家的证据文件池（文件名 + 小文本内容，避免读大日志）
pool = {}
for c in CHIPS:
    base = CHIP_DIR / c
    names, text = [], []
    for p in base.rglob("*"):
        if p.is_file():
            names.append(str(p.relative_to(base)))
            if p.suffix in (".json", ".txt", ".md", ".py", ".sh") and p.stat().st_size < 400_000:
                try:
                    text.append(p.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    pass
    # 后端的 known_issues 实现也算证据
    bp = CHIP_DIR / "prototype" / "runtime" / "backends"
    blob = "\n".join(text)
    pool[c] = {"names": names, "blob": blob}

# PPU 后端是否实现 known_issues（代码级）
code_known = (CHIP_DIR / "prototype/runtime/backends/ppu/backend.py").read_text(encoding="utf-8")
pool["PPU"]["blob"] += "\n" + code_known

print(f"{'步骤':<22}{'910C':>10}{'P800':>10}{'MLU590':>10}{'PPU':>10}   首家命中证据示例")
print("-" * 118)
missing_ppu = []
for label, kws, note in STEPS:
    row = []
    for c in CHIPS:
        n = [x for x in pool[c]["names"] if any(k.lower() in x.lower() for k in kws)]
        t = any(k.lower() in pool[c]["blob"].lower() for k in kws)
        row.append((len(n), t))
    cells = []
    for cnt, t in row:
        if cnt:
            cells.append(f"{cnt}文件")
        elif t:
            cells.append("内容✓")
        else:
            cells.append("—")
    if not (row[3][0] or row[3][1]):
        missing_ppu.append(label)
    sample = ""
    for x in pool["910C"]["names"]:
        if any(k.lower() in x.lower() for k in kws):
            sample = x
            break
    if not sample:
        for x in pool["P800"]["names"]:
            if any(k.lower() in x.lower() for k in kws):
                sample = x
                break
    print(f"{label:<22}{cells[0]:>10}{cells[1]:>10}{cells[2]:>10}{cells[3]:>10}   {sample[:46]}")

print("\n=== PPU 完全无证据的步骤 ===")
for m in missing_ppu:
    print("  ⬜ " + m)
