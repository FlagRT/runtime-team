#!/usr/bin/env python3
"""推理腿 · vLLM 服务化形态验证（基于统一原型，验收标准 2）。

与 proto_infer_leg.py（transformers 前向形态）互补：
  - 本脚本验证**服务化形态**（vLLM OpenAI 兼容接口：/v1/embeddings）
  - 设备上下文仍经统一原型接入（runtime.use / set_device / 流 / 错误翻译）

判据：
  1. 设备上下文可用（统一 API）
  2. 服务健康 + 模型就位
  3. 向量正确（维度、无 NaN/Inf、归一化）
  4. 语义可区分（同主题余弦 > 异主题，区分度 ≥ 0.10）
  5. 吞吐与时延可测（多轮取中位）
  6. 错误注入（超长输入）→ 服务拒绝且可恢复，业务继续
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))          # prototype/runtime/proto
_RUNTIME_DIR = os.path.dirname(_HERE)                        # prototype/runtime
_PKG_DIR = os.path.dirname(_RUNTIME_DIR)                     # prototype
sys.path.insert(0, _HERE)
sys.path.insert(0, _RUNTIME_DIR)
sys.path.insert(0, _PKG_DIR)

import runtime  # noqa: E402

HOST = os.environ.get("SERVE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SERVE_PORT", "8100"))
BASE = f"http://{HOST}:{PORT}"


def http_get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def http_post(path, payload, timeout=120):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:2000]   # 需完整错误体：截断过短会导致 JSON 解析失败、message 丢失
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}


def cosine(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return s / (na * nb) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ascend")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {"mode": "vllm_serve", "backend": args.backend, "checks": {}, "perf": {}}

    # ── 1) 设备上下文（统一原型）──
    runtime.use(args.backend)
    b = runtime.current()
    runtime.set_device(0)
    n_dev = runtime.device_count()
    res["checks"]["device_context"] = {
        "ok": n_dev >= 1,
        "detail": f"后端={b.name} 设备数={n_dev} 已绑定 device 0"}
    print(f"[1] 设备上下文: 后端={b.name} 设备数={n_dev}")

    # 多流：跨流可见性（与服务同卡共存下仍正确）
    try:
        import torch
        dev = f"{b.device_type}:0"
        s = runtime.create_stream()
        x = torch.ones(64, 64, device=dev)
        with b.stream_context(s.native):
            y = x * 3
        b.synchronize(0)
        ok_stream = abs(y.mean().item() - 3.0) < 1e-4
        res["checks"]["stream_on_serving_device"] = {
            "ok": ok_stream, "detail": f"服务运行同卡上跨流计算 = {y.mean().item()}"}
        print(f"[1b] 同卡跨流计算: {y.mean().item()} {'✅' if ok_stream else '❌'}")
    except Exception as e:
        res["checks"]["stream_on_serving_device"] = {
            "ok": False, "detail": f"{type(e).__name__}: {str(e)[:100]}"}

    # ── 2) 服务健康与模型 ──
    st, models = http_get("/v1/models")
    model_id = models["data"][0]["id"] if st == 200 and models.get("data") else None
    res["checks"]["service_ready"] = {
        "ok": st == 200 and model_id is not None,
        "detail": f"/v1/models -> {st}, served model = {model_id}"}
    print(f"[2] 服务就绪: /v1/models {st}, model={model_id}")
    if model_id is None:
        res["verdict"] = "FAIL"
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return

    # ── 3~5) 向量正确性 / 语义区分 / 吞吐 ──
    SAME = ["如何申请退款", "退款流程怎么走", "我要退货"]
    DIFF = ["今天天气怎么样", "推荐一部科幻电影", "怎么做红烧肉"]

    def embed(texts):
        st2, r = http_post("/v1/embeddings", {"model": model_id, "input": texts})
        if st2 != 200:
            raise RuntimeError(f"HTTP {st2}: {str(r)[:160]}")
        return [d["embedding"] for d in r["data"]]

    vecs = embed(SAME + DIFF)
    dim = len(vecs[0])
    finite = all(all(math.isfinite(v) for v in vec) for vec in vecs)
    norm0 = math.sqrt(sum(v * v for v in vecs[0]))
    res["checks"]["embedding_valid"] = {
        "ok": dim > 0 and finite and abs(norm0 - 1.0) < 1e-2,
        "detail": f"维度={dim} 有限={finite} 首条范数={norm0:.6f}"}
    print(f"[3] 向量: 维度={dim} 有限={finite} 范数={norm0:.6f}")

    s_same = (cosine(vecs[0], vecs[1]) + cosine(vecs[0], vecs[2])) / 2
    s_diff = min(cosine(vecs[0], vecs[i]) for i in (3, 4, 5))
    gap = s_same - s_diff
    res["checks"]["semantic_order"] = {
        "ok": s_same > s_diff,
        "detail": f"同主题 {s_same:.4f} > 异主题最小 {s_diff:.4f}"}
    res["checks"]["semantic_gap"] = {
        "ok": gap >= 0.10,
        "detail": f"区分度 {gap:.4f}（判据 ≥0.10）"}
    print(f"[4] 语义: 同主题 {s_same:.4f} / 异主题 {s_diff:.4f} / 区分度 {gap:.4f}")

    # 吞吐与时延
    lat = []
    t0 = time.time()
    for _ in range(args.rounds):
        t1 = time.time()
        embed(SAME)
        lat.append((time.time() - t1) * 1000)
    qps = args.rounds * len(SAME) / (time.time() - t0)
    lat.sort()
    res["perf"] = {
        "qps": round(qps, 2),
        "latency_ms_p50": round(lat[len(lat) // 2], 1),
        "latency_ms_max": round(lat[-1], 1),
        "rounds": args.rounds, "batch": len(SAME)}
    res["checks"]["throughput"] = {"ok": qps > 0, "detail": f"{qps:.2f} 句/s，p50 {lat[len(lat)//2]:.1f}ms"}
    print(f"[5] 吞吐: {qps:.2f} 句/s | p50 {lat[len(lat)//2]:.1f}ms | max {lat[-1]:.1f}ms")

    # ── 6) 错误注入：超长输入 → 服务应拒绝且可恢复 ──
    long_text = "退款" * 6000          # 远超 max-model-len 4096
    err_msg = ""
    try:
        st3, r3 = http_post("/v1/embeddings", {"model": model_id, "input": [long_text]})
        rejected = st3 >= 400
        detail = f"HTTP {st3}"
        # 取服务错误体（含 vLLM 错误类型与消息），用于统一分级
        err_msg = ((r3 or {}).get("error") or {}).get("message", "") if isinstance(r3, dict) else ""
        if err_msg:
            detail += f" | {err_msg[:120]}"
        if st3 == 200:
            detail += "（未拒绝，如实标注）"
    except Exception as e:
        rejected, st3, detail = True, 0, f"请求异常 {type(e).__name__}"

    # 用统一原型对该错误做分级（设备侧职责：识别）
    # 传完整消息（含 vLLM 错误类型），否则分级会退化到保守兜底
    probe = RuntimeError(err_msg or detail)
    fe = runtime.translate_error(probe, location="vllm_serve/embeddings")
    cat = getattr(fe.category, "value", fe.category)
    disp = getattr(fe, "disposition", None)
    res["checks"]["error_injection"] = {
        "ok": rejected,
        "detail": f"{detail}；统一分级 {cat} / disposition={disp}"}
    # 一致性判据：客户端参数错误（HTTP 400）应判 L2_PARAM/raise，而非 L3/replay
    res["checks"]["error_grading_consistency"] = {
        "ok": (cat == "L2_PARAM" and disp == "raise"),
        "detail": f"期望 L2_PARAM/raise（参数类上抛，重试无意义），实测 {cat}/{disp}"}
    print(f"[6] 错误注入: {detail} → {getattr(fe.category, 'value', fe.category)}"
          f" / {getattr(fe, 'disposition', None)}")

    # 业务继续
    try:
        v = embed(SAME[:1])
        ok_continue = len(v[0]) == dim
        res["checks"]["business_continues"] = {
            "ok": ok_continue, "detail": "注入后仍可正常请求，维度一致"}
    except Exception as e:
        res["checks"]["business_continues"] = {
            "ok": False, "detail": f"{type(e).__name__}: {str(e)[:100]}"}
    print(f"[7] 业务继续: {'✅' if res['checks']['business_continues']['ok'] else '❌'}")

    checks = res["checks"]
    res["passed"] = sum(1 for v in checks.values() if v["ok"])
    res["total"] = len(checks)
    res["verdict"] = "SERVE_LEG_PASS" if res["passed"] == res["total"] else "SERVE_LEG_FAIL"
    print(f"\n=== {res['verdict']} | {res['passed']}/{res['total']} ===")

    out = args.out or "proto_infer_serve_result.json"
    with open(os.path.join(_HERE, out), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
