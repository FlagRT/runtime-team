"""单变量隔离探针：每个用例独立进程，避免 CUDA 错误粘滞污染后续用例"""
import json, subprocess, sys, os

CASES = {
 "1_priority_range": "import torch;print('range=',torch.cuda.Stream.priority_range())",
 "2_event_timing": """
import torch,time
s=torch.cuda.Stream();e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
with torch.cuda.stream(s):
    a=torch.randn(4096,4096,device='cuda');b=torch.randn(4096,4096,device='cuda')
    e0.record(s);c=a@b;e1.record(s)
torch.cuda.synchronize()
print('elapsed_ms=%.3f sum=%.1f'%(e0.elapsed_time(e1), c.sum().item()))
""",
 "3_oom_large_alloc": "import torch;torch.empty(int(4e12),dtype=torch.float32,device='cuda')",
 "4_bad_device_ordinal": "import torch;torch.cuda.set_device(99)",
 "5_reset_via_empty_cache": "import torch;torch.cuda.empty_cache();print('empty_cache OK')",
 "6_device_mem_stats": "import torch;print(torch.cuda.mem_get_info(0))",
}

out = {}
for name, code in CASES.items():
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    stdout = r.stdout.strip()
    stderr = r.stderr.strip()
    # 抽取昆仑芯真实错误码
    err_code = None
    for line in stderr.splitlines():
        if "error code=" in line:
            err_code = line.strip()
    out[name] = {
        "rc": r.returncode,
        "stdout": stdout[-300:] if stdout else "",
        "exc": next((l for l in stderr.splitlines() if "Error" in l or "error" in l), "")[:200],
        "kunlun_err_code": err_code,
    }
print(json.dumps(out, ensure_ascii=False, indent=1))
