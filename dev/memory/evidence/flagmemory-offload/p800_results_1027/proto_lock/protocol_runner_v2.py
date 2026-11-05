#!/usr/bin/env python3
# protocol_runner_v2 —— 2026-10-27/28 64GiB 锁定档全腿执行器（PLAN_规范验证跑_真模型_20261027 §3）
# 每引擎一次调用跑全链：P → calib → S1(256/1024/3072 ×3轮冷prompt) → S2(暖1测3) →
# B(c28/57/102/125 各2轮) → L(B110后同引擎史重放2轮) → cancel(R腿资源归还检查)。
# 在宿主机跑（host python3，stdlib only）；通过 docker exec 读容器真值
# （mem_get_info / 引擎进程 RSS / DMA marker 文件）。
# 原始数据一腿一 JSON，另有 chain_status.json 供短轮询。
import argparse, hashlib, json, os, re, subprocess, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = None
OUT = None
TAG = "OFF"
LOG_PATH = None
CTR = "mem-profile-p800"
CONTAINER_OUT = None  # /work 视角输出目录（日志回写用）

B_LEVELS = [28, 57, 102, 125]  # 由池 466,032 tok ÷ 4096 × 25/50/90/110% 推导（round）
S1_LENGTHS = [256, 1024, 3072]

def http_json(path, payload=None, timeout=30):
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def health_ok():
    with urllib.request.urlopen(BASE + "/health", timeout=10) as r:
        return r.status == 200

def make_prompt(i, n_words):
    words = [str(i), str(10000 + i)] + [str((i * 37 + j * 7) % 100) for j in range(n_words)]
    return ' '.join(words[:n_words])

def sha1(s):
    return hashlib.sha1(s.encode()).hexdigest()

def status(leg, extra=None):
    d = {"ts": time.time(), "iso": time.strftime("%m-%d %H:%M:%S"), "tag": TAG, "last_leg": leg}
    if extra:
        d.update(extra)
    p = os.path.join(OUT, "chain_status.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, p)
    print("[STATUS]", json.dumps(d), flush=True)

def docker_exec(script, timeout=300):
    r = subprocess.run(["docker", "exec", CTR, "bash", "-c", script],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout

def meminfo():
    out = docker_exec(
        "source /root/miniconda/bin/activate python310_torch29_cuda && "
        "python /work/p800-1027/probe_meminfo.py 2>/dev/null | tail -1", timeout=240)
    m = re.search(r"free_GiB=([\d.]+) total_GiB=([\d.]+)", out)
    if not m:
        return {"error": out[-200:]}
    return {"free_GiB": float(m.group(1)), "total_GiB": float(m.group(2))}

def rss_all():
    out = docker_exec("bash /work/p800-1027/probe_rss.sh 2>/dev/null")
    procs = []
    for line in out.splitlines():
        m = re.match(r"PID (\d+) RSS_kB (\S+)(?: COMM \S+)? CMD (.*)", line)
        if m:
            procs.append({"pid": int(m.group(1)), "rss_kB": int(float(m.group(2))),
                          "cmd": m.group(3)[:80]})
    procs.sort(key=lambda p: -p["rss_kB"])
    return {"procs": procs[:6], "max_rss_kB": procs[0]["rss_kB"] if procs else None}

MARKER = "/tmp/w2x_marker_transfers.txt"
def marker_read():
    out = docker_exec(f"cat {MARKER} 2>/dev/null || true")
    return out

def marker_parse(text):
    st_b = st_blk = ld_b = ld_blk = 0
    for line in text.splitlines():
        m = re.search(r"\b(store|load) n_blocks=(\d+)", line)
        if not m:
            continue
        if m.group(1) == "store":
            st_b += 1; st_blk += int(m.group(2))
        else:
            ld_b += 1; ld_blk += int(m.group(2))
    return {"store_batches": st_b, "store_blocks": st_blk,
            "load_batches": ld_b, "load_blocks": ld_blk}

def marker_delta(before_text):
    after = marker_read()
    n_before = len(before_text.splitlines())
    lines = after.splitlines()[n_before:]
    return marker_parse("\n".join(lines)), n_before, len(after.splitlines())

class MetricsPoller(threading.Thread):
    FIELDS = ("kv_cache_usage_perc", "num_requests_waiting", "num_requests_running")
    def __init__(self):
        super().__init__(daemon=True)
        self.peaks = {f: 0.0 for f in self.FIELDS}
        self.capacity_peak = 0.0
        self.preemptions_peak = 0.0
        self.stop_flag = False
    def run(self):
        rx_cap = re.compile(r'num_requests_waiting_by_reason.*reason="capacity"\} (\S+)')
        rx_pre = re.compile(r'vllm:num_preemptions_total\{.*?\} (\S+)')
        rx = {f: re.compile(rf'vllm:{f}{{.*?}} (\S+)') for f in self.FIELDS}
        while not self.stop_flag:
            try:
                with urllib.request.urlopen(BASE + "/metrics", timeout=5) as r:
                    txt = r.read().decode("utf-8", "replace")
                for f, rxf in rx.items():
                    for m in rxf.finditer(txt):
                        self.peaks[f] = max(self.peaks[f], float(m.group(1)))
                for m in rx_cap.finditer(txt):
                    self.capacity_peak = max(self.capacity_peak, float(m.group(1)))
                for m in rx_pre.finditer(txt):
                    self.preemptions_peak = max(self.preemptions_peak, float(m.group(1)))
            except Exception:
                pass
            time.sleep(5)

def stream_one(i, n_words, max_tokens, timeout=3600):
    prompt = make_prompt(i, n_words)
    body = json.dumps({"model": "onerec", "prompt": prompt, "max_tokens": max_tokens,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    rec = {"i": i, "t0": time.time(), "ttft": None, "end": None, "usage": None,
           "finish": None, "text_sha1": None, "text_head": None, "error": None}
    chunks = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    rec["usage"] = obj["usage"]
                chs = obj.get("choices") or [{}]
                ch = chs[0]
                txt = ch.get("text")
                if txt:
                    chunks.append(txt)
                    if rec["ttft"] is None:
                        rec["ttft"] = time.time() - rec["t0"]
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
        full = "".join(chunks)
        rec["text_sha1"] = sha1(full)
        rec["text_head"] = full[:80]
        rec["end"] = time.time()
    except Exception as e:
        rec["error"] = repr(e)[:300]
        rec["end"] = time.time()
    return rec

def tokenize_count(prompt):
    try:
        return http_json("/tokenize", {"model": "onerec", "prompt": prompt})["count"]
    except Exception:
        return None

FACTOR = None
def load_factor():
    global FACTOR
    if FACTOR is None:
        with open(os.path.join(OUT, "calib.json")) as f:
            FACTOR = json.load(f)["tok_per_word"]
    return FACTOR

def words_for(target_tok):
    factor = load_factor()
    n = max(2, round(target_tok / factor))
    tok = tokenize_count(make_prompt(0, n))
    dev = abs(tok - target_tok) / target_tok
    if dev > 0.10:  # 一次自动修正
        n = max(2, round(n * target_tok / tok))
        tok = tokenize_count(make_prompt(0, n))
        dev = abs(tok - target_tok) / target_tok
    return n, tok, dev

def save_record(name, obj):
    path = os.path.join(OUT, name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    print("WROTE", path, flush=True)

# ---------- 腿 ----------

def run_P():
    samples = []
    for k, wait in enumerate([0, 150, 300]):  # ≈+0/+2.5/+5min（T0=runner 启动=就绪后）
        if wait:
            time.sleep(wait)
        mi = meminfo()
        samples.append({"t_off_s": wait, "meminfo": mi})
        print(f"P sample {k}: {mi}", flush=True)
    pool_line = conc_line = None
    try:
        with open(LOG_PATH, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", txt)
        if m:
            pool_line = m.group(0)
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", txt)
        if m:
            conc_line = m.group(0)
    except Exception as e:
        pool_line = f"READ_FAIL {e!r}"[:200]
    pool_tok = int(pool_line.split(": ")[1].split(" tokens")[0].replace(",", "")) if pool_line and "GPU KV" in pool_line else None
    bsz = None
    if pool_tok:
        bsz = round(68719476736 / pool_tok)  # bytes/tok 交叉验证
    out = {"leg": "P", "tag": TAG, "pool_log_line": pool_line, "pool_tok": pool_tok,
           "bytes_per_tok": bsz, "concurrency_log_line": conc_line,
           "steady_hbm_samples": samples, "note": "xpu-smi 仅参考不作判据"}
    save_record(f"P-{TAG}.json", out)
    status("P", {"pool_tok": pool_tok})
    return pool_tok

def run_calib():
    n = 512
    prompt = make_prompt(0, n)
    tok = tokenize_count(prompt)
    save_record("calib.json", {"words": n, "tok": tok, "tok_per_word": tok / n})
    status("calib", {"tok_per_word": tok / n})

def run_S1(length, rnd):
    n_words, tok, dev = words_for(length)
    poll = MetricsPoller(); poll.start()
    rec = stream_one(rnd * 7919 + length, n_words, 1)
    poll.stop_flag = True; poll.join()
    ct = (rec["usage"] or {}).get("completion_tokens")
    valid = (ct == 1) and (dev or 0) <= 0.10
    out = {"leg": "S1", "tag": TAG, "round": rnd, "target_tok": length, "words": n_words,
           "actual_prompt_tok": tok, "dev": round(dev, 3), "record": rec,
           "peaks": poll.peaks, "valid": valid,
           "ttft_s": round(rec["ttft"], 2) if rec["ttft"] else None}
    save_record(f"S1-{length}tok-r{rnd}-{TAG}.json", out)
    status(f"S1-{length}-r{rnd}", {"valid": valid, "ttft_s": out["ttft_s"]})

def run_S2(rnd):
    n_words, tok, dev = words_for(256)
    poll = MetricsPoller(); poll.start()
    rec = stream_one(0, n_words, 512)
    poll.stop_flag = True; poll.join()
    ct = (rec["usage"] or {}).get("completion_tokens", 0)
    wall = (rec["end"] - rec["t0"]) if rec["end"] else None
    out = {"leg": "S2", "tag": TAG, "round": rnd, "target_tok": 256, "words": n_words,
           "actual_prompt_tok": tok, "dev": round(dev, 3), "record": rec,
           "peaks": poll.peaks,
           "completion_tokens": ct,
           "valid_decode": ct >= 128 and (dev or 0) <= 0.10,
           "decode_tok_s": round(ct / wall, 2) if wall and ct else None,
           "wall_s": round(wall, 1) if wall else None,
           "chunked_prefill_note": "enable_chunked_prefill=True, prompt<<budget 无交错"}
    save_record(f"S2-r{rnd}-{TAG}.json", out)
    status(f"S2-r{rnd}", {"decode_tok_s": out["decode_tok_s"], "ctok": ct})

B_WORDS = None
def b_words():
    global B_WORDS
    if B_WORDS is None:
        n, tok, dev = words_for(3840)  # 4096−max_tokens128−余量128；余量条款=判据
        assert tok + 128 <= 4096 - 100, f"B prompt margin violated: {tok}+128"
        B_WORDS = (n, tok, dev)
    return B_WORDS

def run_B(c, rnd):
    n, tok, dev = b_words()
    print(f"B leg c={c} r={rnd} words={n} prompt_tok={tok} dev={dev}", flush=True)
    mk_before = marker_read()
    mi_before = meminfo()
    poll = MetricsPoller(); poll.start()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        recs = list(ex.map(lambda i: stream_one(i, n, 128), range(c)))
    wall = time.time() - t0
    poll.stop_flag = True; poll.join()
    delta, n_before, n_after = marker_delta(mk_before)
    ok = [r for r in recs if r["error"] is None]
    ptok = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in ok)
    ctok = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    e2e = sorted(r["end"] - r["t0"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    agg = ctok / wall if wall else None
    margin_ok = all((r["usage"] or {}).get("prompt_tokens", 10**9) + 128 <= 4096 for r in ok)
    out = {"leg": "B", "tag": TAG, "c": c, "round": rnd, "words": n,
           "target_prompt_tok": 3840, "actual_prompt_tok": tok, "dev": round(dev, 3),
           "margin_ok": margin_ok,
           "n_req": c, "ok": len(ok), "failed": len(recs) - len(ok),
           "wall_s": round(wall, 1), "e2e_p50_s": round(e2e[len(e2e)//2], 1) if e2e else None,
           "e2e_max_s": round(e2e[-1], 1) if e2e else None,
           "ttft_p50_s": round(ttfts[len(ttfts)//2], 2) if ttfts else None,
           "ttft_max_s": round(ttfts[-1], 2) if ttfts else None,
           "sum_prompt_tok": ptok, "sum_completion_tok": ctok,
           "agg_decode_tok_s": round(agg, 1) if agg else None,
           "peaks": poll.peaks, "capacity_waiting_peak": poll.capacity_peak,
           "preemptions_peak": poll.preemptions_peak,
           "dma_delta": delta, "marker_lines_before": n_before, "marker_lines_after": n_after,
           "meminfo_before": mi_before,
           "records": recs}
    save_record(f"B-c{c}-r{rnd}-{TAG}.json", out)
    status(f"B-c{c}-r{rnd}", {"wall_s": out["wall_s"], "ok": len(ok),
                              "capW": poll.capacity_peak,
                              "dma": {k: delta[k] for k in ("store_batches", "load_batches")}})

def run_L(c):
    """B-c125-r2 完成后立即调用：同引擎史内重放该档唯一 prompt 批 2 轮。"""
    n, tok, dev = b_words()
    # 同引擎 reference = B-c125 r2（与 r1）输出 sha1
    ref = {}
    for rnd in (1, 2):
        p = os.path.join(OUT, f"B-c{c}-r{rnd}-{TAG}.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        ref[rnd] = {r["i"]: r.get("text_sha1") for r in d["records"] if r["error"] is None}
    mi_before = meminfo()
    rss_before = rss_all()
    mk_before = marker_read()
    rounds = []
    for rnd in (1, 2):
        poll = MetricsPoller(); poll.start()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            recs = list(ex.map(lambda i: stream_one(i, n, 128), range(c)))
        wall = time.time() - t0
        poll.stop_flag = True; poll.join()
        ok = [r for r in recs if r["error"] is None]
        eq_r2 = sum(1 for r in ok if r.get("text_sha1") and r["text_sha1"] == ref[2].get(r["i"]))
        eq_r1 = sum(1 for r in ok if r.get("text_sha1") and r["text_sha1"] == ref[1].get(r["i"]))
        rounds.append({"round": rnd, "wall_s": round(wall, 1), "ok": len(ok),
                       "eq_vs_B_r2": eq_r2, "eq_vs_B_r1": eq_r1,
                       "n": len(ok), "peaks": poll.peaks,
                       "capacity_waiting_peak": poll.capacity_peak,
                       "records": recs})
        status(f"L-r{rnd}", {"wall_s": round(wall, 1), "eq_r2": eq_r2})
    delta, n_before, n_after = marker_delta(mk_before)
    mi_after = meminfo()
    rss_after = rss_all()
    out = {"leg": "L", "tag": TAG, "c": c, "words": n, "actual_prompt_tok": tok,
           "ref_source": f"B-c{c}-r1/r2 same-engine",
           "rounds": rounds,
           "dma_delta": delta, "marker_lines_before": n_before, "marker_lines_after": n_after,
           "meminfo_before": mi_before, "meminfo_after": mi_after,
           "rss_before": rss_before, "rss_after": rss_after,
           "load_criterion": {"load_batches_gt0": delta["load_batches"] > 0,
                              "store_batches": delta["store_batches"],
                              "load_batches": delta["load_batches"],
                              "load_blocks": delta["load_blocks"]}}
    save_record(f"L-c{c}-{TAG}.json", out)
    status("L", {"load_batches": delta["load_batches"], "store_batches": delta["store_batches"]})

def run_cancel():
    """R 腿资源归还检查：流式请求首 chunk 后掐断 → 静置 → meminfo 归位 + 后续请求可用。"""
    n2, tok2, dev2 = words_for(256)
    mi_before = meminfo()
    prompt = make_prompt(77000, n2)  # 冷 prompt
    body = json.dumps({"model": "onerec", "prompt": prompt, "max_tokens": 512,
                       "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; got = 0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                txt = (obj.get("choices") or [{}])[0].get("text")
                if txt:
                    got += 1
                    if ttft is None:
                        ttft = time.time() - t0
                        break  # 掐断（with 退出即断连 → 客户端 abort）
    except Exception as e:
        pass
    time.sleep(10)  # 引擎处理 abort
    mi_after = meminfo()
    # 归还判据：free 回到 cancel 前 ±0.3 GiB，且后续请求成功
    follow = stream_one(77001, 64, 8, timeout=300)
    follow_ok = follow["error"] is None and (follow["usage"] or {}).get("completion_tokens") == 8
    out = {"leg": "R-cancel", "tag": TAG, "ttft_s": round(ttft, 2) if ttft else None,
           "chunks_before_abort": got,
           "meminfo_before": mi_before, "meminfo_after_abort": mi_after,
           "returned": abs(mi_before.get("free_GiB", 0) - mi_after.get("free_GiB", 0)) <= 0.30,
           "followup_ok": follow_ok, "followup_record": {k: follow[k] for k in ("error", "usage")},
           "prompt_tok": tok2}
    save_record(f"R-cancel-{TAG}.json", out)
    status("R-cancel", {"returned": out["returned"], "followup_ok": follow_ok})

def main():
    global BASE, OUT, TAG, LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["ON", "OFF"])
    ap.add_argument("--port", type=int, default=18321)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--skip-p", action="store_true")
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"
    OUT = args.out
    TAG = args.tag
    LOG_PATH = args.log
    os.makedirs(OUT, exist_ok=True)
    health_ok()
    print(f"CHAIN START tag={TAG} base={BASE} out={OUT} log={LOG_PATH}", flush=True)
    status("start")
    if not args.skip_p:
        run_P()
    run_calib()
    for L in S1_LENGTHS:
        for r in (1, 2, 3):
            try:
                run_S1(L, r)
            except Exception as e:
                status(f"S1-{L}-r{r}-ERR", {"error": repr(e)[:200]})
    try:
        run_S2(1)
    except Exception as e:
        status(f"S2-r1-ERR", {"error": repr(e)[:200]})
    # 污染金丝雀：S2 首轮应 ≥8 tok/s（先例 OFF 13.90 / 本跑 ON 13.37；共卡污染形态=1.7）。
    if TAG == "OFF":
        try:
            s2r1 = json.load(open(os.path.join(OUT, f"S2-r1-{TAG}.json")))
            if (s2r1.get("decode_tok_s") or 0) < 8:
                status("CONTAMINATION_ABORT", {"s2_r1_tok_s": s2r1.get("decode_tok_s"),
                                               "reason": "decode 8x 慢 = 共卡污染形态，链中止避免污染全链"})
                print("CONTAMINATION_ABORT", flush=True)
                return
        except Exception:
            pass
    for r in (2, 3, 4):
        try:
            run_S2(r)
        except Exception as e:
            status(f"S2-r{r}-ERR", {"error": repr(e)[:200]})
    for c in B_LEVELS:
        for r in (1, 2):
            try:
                run_B(c, r)
            except Exception as e:
                status(f"B-c{c}-r{r}-ERR", {"error": repr(e)[:200]})
    try:
        run_L(125)
    except Exception as e:
        status("L-ERR", {"error": repr(e)[:200]})
    try:
        run_cancel()
    except Exception as e:
        status("R-cancel-ERR", {"error": repr(e)[:200]})
    # DMA 总账（全窗口累计）
    total = marker_parse(marker_read())
    save_record(f"dma_ledger-{TAG}.json",
                {"tag": TAG, "total": total,
                 "note": "分段账见各腿 JSON 的 dma_delta；store/load 分开计"})
    status("DONE", {"dma_total": total})
    print("CHAIN DONE", flush=True)

if __name__ == "__main__":
    main()
