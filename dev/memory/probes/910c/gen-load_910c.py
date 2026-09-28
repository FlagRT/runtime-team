#!/usr/bin/env python3
"""生成式请求驱动 + vLLM /metrics 轮询采集(910C 显存画像)。

用法(容器内或宿主,需能访问服务端口):
  python3 gen-load_910c.py --port 8107 --model qwen3-4b       --prompt-tokens 128 --max-tokens 256 --concurrency 1 --rounds 1       --tag c-out256 --out /workspace/dev/memory/benchmarks/out/gen910c

行为:每轮并发发 N 个 /v1/completions 请求(忽略 eos,用 ignore_eos 强制生成满
max_tokens),同时后台线程每秒轮询 /metrics,记录 kv_cache_usage_perc、
num_requests_running/waiting、累计 prompt/generation tokens。结果落
<tag>_load.json + <tag>_kv.csv。token 数取响应 usage 实测值。
"""
import argparse, csv, json, time, threading, urllib.request, sys

def scrape(base):
    out = {}
    try:
        with urllib.request.urlopen(base + '/metrics', timeout=5) as r:
            for line in r.read().decode().splitlines():
                if line.startswith('#') or '{' in line.split(' ')[0]:
                    pass
                parts = line.rsplit(' ', 1)
                if len(parts) != 2:
                    continue
                name, val = parts[0], parts[1]
                for k in ('vllm:kv_cache_usage_perc', 'vllm:num_requests_running',
                          'vllm:num_requests_waiting', 'vllm:prompt_tokens_total',
                          'vllm:generation_tokens_total', 'vllm:prefix_cache_hits_total'):
                    if name.startswith(k):
                        try:
                            out[k] = float(val)
                        except ValueError:
                            pass
    except Exception as e:
        out['error'] = str(e)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8107)
    p.add_argument('--model', required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--prompt-tokens', type=int, default=128)
    p.add_argument('--tokenizer-path', default=None,
                   help='提供时用 transformers tokenizer 精确构造输入长度(推荐)')
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--cancel-after', type=float, default=0,
                   help='>0 时每请求发出去这么多秒后断开连接(测取消/归还)')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--tag', required=True)
    p.add_argument('--out', default='.')
    a = p.parse_args()
    base = f'http://{a.host}:{a.port}'

    if a.tokenizer_path:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer_path)
        # 以重复数字串为填充,截/补到精确长度
        fill = tok.encode('回顾序列 1 2 3 4 5 6 7 8 9 ' * 2000)
        ids = fill[:a.prompt_tokens]
        prompt = tok.decode(ids)
        n = len(tok.encode(prompt))
        while n < a.prompt_tokens:
            prompt += ' 1'
            n = len(tok.encode(prompt))
        print(f'exact prompt built: target={a.prompt_tokens} actual={n}', file=sys.stderr)
    else:
        prompt = '数据 ' * max(1, a.prompt_tokens // 2)  # 粗略;实测以 usage 为准

    def one_request(idx):
        body = json.dumps({
            'model': a.model, 'prompt': prompt + f' [req{idx}]',
            'max_tokens': a.max_tokens, 'ignore_eos': True, 'temperature': 0,
            'stream': False,
        }).encode()
        req = urllib.request.Request(base + '/v1/completions', data=body,
                                     headers={'Content-Type': 'application/json'})
        t0 = time.time()
        try:
            if a.cancel_after > 0:
                # 简易取消:读超时后断开,客户端放弃响应
                with urllib.request.urlopen(req, timeout=a.cancel_after) as r:
                    r.read()
                rec = {'idx': idx, 'status': 'ok-unexpected'}
            else:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    resp = json.loads(r.read())
                u = resp.get('usage', {})
                rec = {'idx': idx, 'status': 'done', 't_s': round(time.time() - t0, 3),
                       'prompt_tokens': u.get('prompt_tokens'),
                       'completion_tokens': u.get('completion_tokens')}
        except Exception as e:
            rec = {'idx': idx, 'status': f'cancelled/{type(e).__name__}',
                   't_s': round(time.time() - t0, 3)}
        return rec

    # warmup(不计入)
    for _ in range(a.warmup):
        try:
            one_request(-1)
        except Exception:
            pass

    kv_rows, stop = [], threading.Event()
    def poller():
        t0 = time.time()
        while not stop.is_set():
            m = scrape(base)
            kv_rows.append({'t': round(time.time() - t0, 1), **m})
            time.sleep(1.0)
    th = threading.Thread(target=poller); th.start()

    results = []
    t_all = time.time()
    for rnd in range(a.rounds):
        ts = [threading.Thread(target=lambda i=i: results.append(
            {'round': rnd, **one_request(i)})) for i in range(a.concurrency)]
        for t in ts: t.start()
        for t in ts: t.join()
        time.sleep(3)  # 让归还后的指标落进采样
    stop.set(); th.join()

    with open(f'{a.out}/{a.tag}_kv.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['t'] + sorted({k for r in kv_rows for k in r if k != 't'}))
        w.writeheader(); w.writerows(kv_rows)
    summary = {'tag': a.tag, 'args': vars(a), 'wall_s': round(time.time() - t_all, 1),
               'requests': results,
               'kv_usage_perc_max': max((r.get('vllm:kv_cache_usage_perc', 0) for r in kv_rows), default=None),
               'kv_usage_perc_last': next((r.get('vllm:kv_cache_usage_perc') for r in reversed(kv_rows)
                                           if 'vllm:kv_cache_usage_perc' in r), None)}
    with open(f'{a.out}/{a.tag}_load.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in ('tag', 'wall_s', 'kv_usage_perc_max',
                                              'kv_usage_perc_last')}, ensure_ascii=False))
    done = [r for r in results if r['status'] == 'done']
    print(f'done {len(done)}/{len(results)}; sample:', json.dumps(done[0] if done else results[:1], ensure_ascii=False))

if __name__ == '__main__':
    main()
