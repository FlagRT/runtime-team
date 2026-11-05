#!/usr/bin/env python3
# PROFILE-PROTOCOL v0.2 执行器（P800 / OneRec-8B，2026-10-25）。
# 每腿一次调用，产出带判据字段的 JSON 记录；metrics 采样线程抓池利用率/waiting 峰值。
# 动作：calib(S1/S2/B 前置) | S1 --length N --round R | S2 --round R | B --c N --round R
import argparse, hashlib, json, os, re, threading, time, urllib.request

BASE = None
OUT = None
TAG = "OFF"

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

class MetricsPoller(threading.Thread):
    FIELDS = ("kv_cache_usage_perc", "num_requests_waiting", "num_requests_running")
    def __init__(self):
        super().__init__(daemon=True)
        self.peaks = {f: 0.0 for f in self.FIELDS}
        self.capacity_peak = 0.0
        self.stop_flag = False
    def run(self):
        rx_cap = re.compile(r'num_requests_waiting_by_reason.*reason="capacity"\} (\S+)')
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
            except Exception:
                pass
            time.sleep(5)

def stream_one(i, n_words, max_tokens):
    """发单请求（流式），返回记录 dict。usage 来自 include_usage 尾块。"""
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
        with urllib.request.urlopen(req, timeout=1800) as r:
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
        rec["text_sha1"] = hashlib.sha1(full.encode()).hexdigest()
        rec["text_head"] = full[:80]
        rec["end"] = time.time()
    except Exception as e:
        rec["error"] = repr(e)[:300]
        rec["end"] = time.time()
    return rec

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

def tokenize_count(prompt):
    try:
        return http_json("/tokenize", {"model": "onerec", "prompt": prompt})["count"]
    except Exception:
        return None

FACTOR_FILE = None
def load_factor():
    with open(FACTOR_FILE) as f:
        return json.load(f)["tok_per_word"]

def save_record(name, obj):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print("WROTE", path)

def run_S1(length, rnd):
    n_words, tok, dev = words_for(length)
    poll = MetricsPoller(); poll.start()
    # 规范漏洞修复（2026-10-25 实跑发现）：prefill 腿每轮必须冷 prompt——
    # 同 prompt 第 2/3 轮是 prefix cache 命中（TTFT 0.1s），测的不是 prefill。
    # i = rnd*7919 + length 保证 轮次x档位 唯一。
    rec = stream_one(rnd * 7919 + length, n_words, 1)
    poll.stop_flag = True; poll.join()
    valid = (rec["usage"] or {}).get("completion_tokens") == 1 and (dev or 0) <= 0.10
    out = {"leg": "S1", "tag": TAG, "round": rnd, "target_tok": length, "words": n_words,
           "actual_prompt_tok": tok, "dev": round(dev, 3), "record": rec,
           "peaks": poll.peaks, "valid": valid,
           "ttft_s": round(rec["ttft"], 2) if rec["ttft"] else None}
    save_record(f"S1-{length}tok-r{rnd}-{TAG}.json", out)

def run_S2(rnd):
    n_words, tok, dev = words_for(256)
    poll = MetricsPoller(); poll.start()
    rec = stream_one(0, n_words, 512)
    poll.stop_flag = True; poll.join()
    ct = (rec["usage"] or {}).get("completion_tokens", 0)
    wall = (rec["end"] - rec["t0"]) if rec["end"] else None
    out = {"leg": "S2", "tag": TAG, "round": rnd, "target_tok": 256, "words": n_words,
           "actual_prompt_tok": tok, "dev": round(dev, 3), "record": rec,
           "peaks": poll.peaks, "valid": ct >= 128 and (dev or 0) <= 0.10,
           "decode_tok_s": round(ct / wall, 2) if wall and ct else None,
           "wall_s": round(wall, 1) if wall else None}
    save_record(f"S2-r{rnd}-{TAG}.json", out)

def run_B(c, rnd, words):
    poll = MetricsPoller(); poll.start()
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=c) as ex:
        recs = list(ex.map(lambda i: stream_one(i, words, 128), range(c)))
    wall = time.time() - t0
    poll.stop_flag = True; poll.join()
    ok = [r for r in recs if r["error"] is None]
    ptok = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in ok)
    ctok = sum((r["usage"] or {}).get("completion_tokens", 0) for r in ok)
    e2e = sorted(r["end"] - r["t0"] for r in ok)
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
    agg = ctok / wall if wall else None
    out = {"leg": "B", "tag": TAG, "c": c, "round": rnd, "words": words,
           "n_req": c, "ok": len(ok), "failed": len(recs) - len(ok),
           "wall_s": round(wall, 1), "e2e_p50_s": round(e2e[len(e2e)//2], 1) if e2e else None,
           "e2e_max_s": round(e2e[-1], 1) if e2e else None,
           "ttft_p50_s": round(ttfts[len(ttfts)//2], 2) if ttfts else None,
           "ttft_max_s": round(ttfts[-1], 2) if ttfts else None,
           "sum_prompt_tok": ptok, "sum_completion_tok": ctok,
           "agg_decode_tok_s": round(agg, 1) if agg else None,
           "peaks": poll.peaks, "capacity_waiting_peak": poll.capacity_peak,
           "records": recs}
    save_record(f"B-c{c}-r{rnd}-{TAG}.json", out)

def run_calib():
    n = 512
    prompt = make_prompt(0, n)
    tok = tokenize_count(prompt)
    save_record("calib.json", {"words": n, "tok": tok, "tok_per_word": tok / n})

def main():
    global BASE, OUT, TAG, FACTOR_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", required=True, choices=["calib", "S1", "S2", "B"])
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--c", type=int, default=28)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--tag", default="OFF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18307)
    args = ap.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"
    OUT = args.out
    TAG = args.tag
    FACTOR_FILE = os.path.join(OUT, "calib.json")
    os.makedirs(OUT, exist_ok=True)
    health_ok()
    if args.action == "calib":
        run_calib()
    elif args.action == "S1":
        run_S1(args.length, args.round)
    elif args.action == "S2":
        run_S2(args.round)
    elif args.action == "B":
        n, tok, dev = words_for(3840)  # 3840+128=3968 < 4096，余量防 header 位数方差顶爆 max_model_len
        print(f"B leg c={args.c} words={n} prompt_tok={tok} dev={dev}")
        run_B(args.c, args.round, n)

if __name__ == "__main__":
    main()
