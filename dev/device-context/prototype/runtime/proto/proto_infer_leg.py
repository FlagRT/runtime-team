"""推理腿自验证（V2，**后端无关**）：经统一运行时 API 跑通向量输出。

验证点（对应验收标准 2 推理腿部分）：
  1. 设备上下文可用（统一 API 绑定设备 / 建流 / 有界同步 / 显存口径）
  2. 模型在设备上加载成功
  3. 向量结果正确（同句相似度≈1、异句明显低、维度/范数合理、无 NaN）
  4. 吞吐与时延数据（含 p50 / p90）
  5. 错误识别：真实参数类异常 → L1–L4 分级 + 处置策略（后端无关）

**后端无关化的由来（2026-09-20，昆仑芯 P800 第二阶段）**
  原实现硬编码 910C 路径、`npu:0` 设备串与 `torch.npu.synchronize()`，只能在昇腾上跑。
  现改为**环境变量驱动**（默认值自适应脚本位置，不再硬编码厂商路径），
  设备串统一取 `runtime.current().device_type`，同步统一走 `runtime.synchronize()`。
  ⇒ 同一份脚本可在两处运行，这正是「统一 API：换芯片只改一行」的体现。

环境变量（括号内为默认值）
  DC_BACKEND   运行时后端名（`ascend`；昆仑芯用 `kunlun`）
  DC_ROOT      device-context 根路径（默认 = 脚本上两级目录，即 prototype/）
  DC_MODEL     模型路径（`/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B`）
  DC_OUT_DIR   结果输出目录（默认 = 脚本所在目录）
  DC_ROUNDS    计时轮数（5）

用法（昆仑芯 P800）：
  # 用卡前先 `xpu-smi` 挑**空闲**的卡；共享机上被他人占用的卡会触发设备侧报错
  # ⚠️ 同训练腿：`DC_MODEL` 必须给到 `snapshots/<hash>`（给缓存根目录会报 Unrecognized model）；
  #    脚本路径是 `runtime/proto/proto_infer_leg.py`。
  CUDA_VISIBLE_DEVICES=6 DC_BACKEND=kunlun \
  DC_ROOT=/workspace/prototype \
  DC_MODEL=/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3 \
  DC_OUT_DIR=/workspace/out_infer \
  python3 runtime/proto/proto_infer_leg.py

用法（910C，保持原口径）：
  DC_BACKEND=ascend python3 proto_infer_leg.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

# ── 后端无关化：路径由环境变量驱动，默认自适应脚本位置 ──
_HERE = os.path.dirname(os.path.abspath(__file__))          # prototype/runtime/proto
_PKG_DIR = os.path.dirname(os.path.dirname(_HERE))          # prototype
ROOT = os.environ.get("DC_ROOT", _PKG_DIR)
sys.path.insert(0, ROOT)

import runtime  # noqa: E402  —— 我们的统一运行时原型

MODEL_RAW = os.environ.get("DC_MODEL", "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B")
OUT_DIR = os.environ.get("DC_OUT_DIR", _HERE)
ROUNDS = int(os.environ.get("DC_ROUNDS", "5"))
BACKEND_DEFAULT = os.environ.get("DC_BACKEND", "ascend")


def resolve_model(path: str) -> str:
    """兼容三种 DC_MODEL 传法，返回 `from_pretrained` 可直接用的路径。

    1. snapshot 路径（含 config.json）        → 原样使用
    2. hub 的 models--xxx 目录（仅含 snapshots/…）→ 自动解析到 snapshots/<hash>
       （P800 的共享缓存挂载在 /hf_cache，训练腿即用此形式）
    3. 任意普通本地目录                        → 原样使用
    """
    if os.path.isdir(path):
        snaps = os.path.join(path, "snapshots")
        if os.path.isdir(snaps):
            hashes = sorted(d for d in os.listdir(snaps)
                            if os.path.isdir(os.path.join(snaps, d)))
            if hashes:
                return os.path.join(snaps, hashes[-1])
    return path


MODEL = resolve_model(MODEL_RAW)

SENTS = [
    "如何申请退款",
    "退款流程是怎样的",
    "今天天气怎么样",
]

INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery:")


def python_version_line():
    import platform
    return f"python {platform.python_version()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=BACKEND_DEFAULT)
    ap.add_argument("--out", default="proto_infer_leg_result.json")
    args = ap.parse_args()

    res = {"model": MODEL, "model_raw": MODEL_RAW, "backend": args.backend, "env": {}, "checks": {}}

    # ── 1. 统一运行时 API 自证（设备无关）──
    b = runtime.use(args.backend)
    dev = f"{b.device_type}:0"                      # ← 由后端决定，不硬编码 npu/cuda
    n = runtime.device_count()
    runtime.set_device(0)
    st = runtime.create_stream()
    mem = runtime.memory_stats(0)
    res["env"] = {
        "python": python_version_line(),
        "backend": args.backend,
        "device_type": b.device_type,
        "device_count": n,
        "capabilities": sorted(b.info().get("capabilities", [])),
    }
    res["checks"]["device_count"] = {"ok": n > 0, "detail": f"可见设备 {n}"}
    res["checks"]["stream"] = {"ok": st is not None, "detail": f"{st!r}"}
    res["checks"]["memory_stats"] = {"ok": bool(mem), "detail": str(mem)[:120]}
    res["checks"]["probe"] = {"ok": bool(runtime.probe_device(0)), "detail": "设备探活"}
    print(f"[1] 设备上下文: backend={args.backend} dev={dev} count={n} mem={str(mem)[:60]}")

    # ── 2. 模型加载 ──
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(dev).eval()
    res["checks"]["model_load"] = {"ok": True, "detail": f"transformers 加载到 {dev}"}
    print(f"[2] 模型加载完成 → {dev}")

    def last_token_pool(hidden, attention_mask):
        # 模型 1_Pooling 配置：pooling_mode_lasttoken = true
        lens = attention_mask.sum(dim=1) - 1                 # 最后一个非 padding token
        idx = lens[:, None, None].expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)

    def encode(texts, rounds=ROUNDS, with_instruct=True):
        inputs = [INSTRUCT + " " + t if with_instruct else t for t in texts]
        for _ in range(2):                                    # 预热
            with torch.no_grad():
                bx = tok(inputs, padding=True, truncation=True, return_tensors="pt").to(dev)
                _ = model(**bx)
        runtime.synchronize(0)                                # ← 统一 API（原来写死 torch.npu）
        lat = []
        outs = None
        for _ in range(rounds):
            t0 = time.time()
            with torch.no_grad():
                bx = tok(inputs, padding=True, truncation=True, return_tensors="pt").to(dev)
                outs = model(**bx)
            runtime.synchronize(0)
            lat.append(time.time() - t0)
        v = last_token_pool(outs.last_hidden_state, bx["attention_mask"])
        # 先在 fp32 下归一化：半精度归一化的范数偏差可达 1e-3 量级（实测 1.56e-3）
        v = torch.nn.functional.normalize(v.float(), dim=-1)
        return v.cpu(), lat

    vecs, lat = encode(SENTS, with_instruct=True)
    vecs_list = vecs.tolist()

    # ── 3. 向量正确性 ──
    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    s_same = cos(vecs_list[0], vecs_list[1])   # 语义相近（退款）
    s_diff = cos(vecs_list[0], vecs_list[2])   # 语义不同（天气）
    norms = [round(math.sqrt(cos(v, v)), 6) for v in vecs_list]
    has_nan = any(math.isnan(x) for v in vecs_list for x in v)
    dim = len(vecs_list[0])
    gap = s_same - s_diff

    res["checks"]["vector_dim"] = {"ok": dim > 0, "detail": f"维度 {dim}（910C 参照 1024）"}
    res["checks"]["vector_normalized"] = {
        "ok": all(abs(x - 1.0) < 1e-3 for x in norms), "detail": f"范数 {norms}"}
    res["checks"]["vector_no_nan"] = {"ok": not has_nan, "detail": "无 NaN"}
    res["checks"]["semantic_order"] = {
        "ok": s_same > s_diff,
        "detail": f"同主题相似度 {s_same:.4f} > 异主题 {s_diff:.4f}"}
    res["checks"]["semantic_gap"] = {
        "ok": gap >= 0.10,
        "detail": f"区分度 {gap:.4f}（判据：≥0.10 视为语义可区分；"
                  f"采用 last-token 池化 + query 指令前缀，符合模型自带 1_Pooling 配置）"}
    print(f"[3] 向量: dim={dim} 同主题 {s_same:.4f} / 异主题 {s_diff:.4f} / 范数 {norms[0]}")

    # ── 4. 性能（含 p50 / p90，对齐 910C 口径）──
    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] * 1000
    p90 = lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.9))] * 1000
    avg = sum(lat) / len(lat)
    res["perf"] = {
        "batch": len(SENTS),
        "rounds": ROUNDS,
        "avg_latency_ms": round(avg * 1000, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "throughput_sent_per_s": round(len(SENTS) / avg, 2),
    }
    print(f"[4] 吞吐: {res['perf']['throughput_sent_per_s']} 句/s, "
          f"p50 {res['perf']['p50_latency_ms']} ms, avg {res['perf']['avg_latency_ms']} ms")

    # ── 5. 错误识别（真实异常注入，后端无关）──
    #  原实现翻译的是一个**伪造的 ACL 错误码字符串**（昇腾专用，跨后端无意义）。
    #  现改为注入**真实参数类异常**，验证「识别 → 分级 → 处置」三段都走通；
    #  再按后端能力决定是否补厂商错误码用例（supports("error_map")）。
    err_checks = {}
    try:
        a = torch.ones(4, 4, device=dev)
        bb = torch.ones(3, 3, device=dev)
        _ = a @ bb                                          # 形状不匹配 → 参数类异常
        err_checks["inject_ok"] = {"ok": False, "detail": "预期抛错但未抛（注入失败）"}
    except Exception as exc:  # noqa: BLE001
        fe = runtime.translate_error(exc, location="infer-leg:op:matmul")
        err_checks["inject_ok"] = {"ok": True, "detail": f"{type(exc).__name__}: {str(exc)[:80]}"}
        err_checks["translate"] = {
            "ok": fe.category is not None and fe.disposition in
                  ("retry", "raise", "replay", "device_recovery"),
            "detail": f"{fe.category.value} → {fe.disposition}（graded_by={fe.graded_by}, "
                      f"confident={fe.is_grade_confident}）",
        }
        # 业务是否继续可用（错误不污染设备）
        try:
            ok_val = float((torch.ones(8, 8, device=dev) * 2).sum().item())
            err_checks["business_continue"] = {"ok": abs(ok_val - 128.0) < 1e-3,
                                               "detail": f"错误后业务继续，校验值 {ok_val}"}
        except Exception as exc2:  # noqa: BLE001
            err_checks["business_continue"] = {"ok": False, "detail": f"业务中断: {exc2}"}
        print(f"[5] 错误分级: {fe.category.value} → {fe.disposition} "
              f"(graded_by={fe.graded_by})")

    if b.supports("error_map"):
        # 厂商码表可用时才跑（如昇腾 ACL 错误码）；不支持则如实标注，不计入失败
        fe2 = runtime.translate_error(
            RuntimeError("ACL stream sync timeout, error code is 507046"), location="infer-leg")
        err_checks["vendor_code_map"] = {
            "ok": fe2.graded_by == "code_map" or fe2.mapped,
            "detail": f"507046 → {fe2.category.value}（graded_by={fe2.graded_by}）",
        }
    else:
        err_checks["vendor_code_map"] = {
            "ok": True, "skipped": True,
            "detail": "该后端不支持 error_map（无厂商错误码透出）——如实跳过，不计入失败",
        }

    res["checks"].update(err_checks)

    # ── 6. 汇总 ──
    counted = {k: v for k, v in res["checks"].items() if not v.get("skipped")}
    ok = all(v["ok"] for v in counted.values())
    res["verdict"] = "INFER_LEG_PASS" if ok else "INFER_LEG_FAIL"
    res["passed"] = sum(1 for v in counted.values() if v["ok"])
    res["total"] = len(counted)
    res["skipped"] = [k for k, v in res["checks"].items() if v.get("skipped")]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"\n[rank0] {res['verdict']} {res['passed']}/{res['total']} | "
          f"backend={args.backend} dev={dev} | dim {dim} | "
          f"{res['perf']['throughput_sent_per_s']} 句/s p50 {res['perf']['p50_latency_ms']}ms "
          f"| 区分度 {gap:.4f} -> {out_path}")
    if res["skipped"]:
        print(f"  （如实跳过：{', '.join(res['skipped'])}）")


if __name__ == "__main__":
    main()
