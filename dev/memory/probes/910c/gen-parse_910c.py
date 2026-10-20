#!/usr/bin/env python3
"""从 vllm_serve.log 抓显存结算行,输出 JSON(A 组加载基线用)。
用法: python3 gen-parse_910c.py <vllm.log> <out.json> [tag]"""
import json, re, sys
log = open(sys.argv[1], encoding='utf-8', errors='replace').read()
out = {'tag': sys.argv[3] if len(sys.argv) > 3 else ''}
def g(pat, cast=float):
    m = re.findall(pat, log)
    return cast(m[-1]) if m else None
out['weights_gib'] = g(r'Loading model weights took ([\d.]+) GB')
out['kv_pool_gib'] = g(r'Available KV cache memory: ([\d.]+) GiB')
out['kv_cache_tokens'] = g(r'GPU KV cache size: ([\d,]+) tokens', lambda x: int(x.replace(',', '')))
m = re.findall(r'Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x', log)
if m:
    out['max_concurrency_ctx_tokens'] = int(m[-1][0].replace(',', ''))
    out['max_concurrency'] = float(m[-1][1])
free = re.findall(r'Free memory on device \(([\d.]+)/([\d.]+) GiB\)[^\n]*Actual usage: ([\d.]+) GiB for weights, ([\d.]+) GiB for peak activation, ([\d.]+) GiB for non-torch memory, ([\d.]+) GiB for NPU graph memory[^\n]*Current KV cache memory: ([\d.]+) GiB', log)
if free:
    f = free[-1]
    out.update({'free_at_startup_gib': float(f[0]), 'total_visible_gib': float(f[1]),
                'weights_gib': float(f[2]), 'peak_activation_gib': float(f[3]),
                'non_torch_gib': float(f[4]), 'npu_graph_gib': float(f[5]),
                'kv_cache_gib': float(f[6])})
out['gpu_mem_util'] = g(r'Desired GPU memory utilization is \(([\d.]+),', float)
out['page_attention_note'] = 'settlement from vllm_serve.log last occurrence'
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False))
