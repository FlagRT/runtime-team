#!/usr/bin/env python3
"""下载 Qwen3-30B-A3B 全部文件（hf-mirror.com 直连，断点续传，并行）。
用法: python3 fetch_qwen3_30b.py <file_list_json> <dest_dir> [parallel]
file_list_json 由 HF tree API 生成，含 path/size。
"""
import json
import os
import subprocess
import sys
import threading
from queue import Queue

BASE = "https://hf-mirror.com/Qwen/Qwen3-30B-A3B/resolve/main"

def main():
    list_json, dest = sys.argv[1], sys.argv[2]
    parallel = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    entries = json.load(open(list_json))
    files = [(e["path"], e["size"]) for e in entries if e.get("type") == "file"]
    os.makedirs(dest, exist_ok=True)
    total = sum(s for _, s in files)
    done_size = 0
    lock = threading.Lock()
    ok, fail = [], []

    def worker():
        nonlocal done_size
        while True:
            item = q.get()
            if item is None:
                return
            path, size = item
            out = os.path.join(dest, path)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            # 已完整则跳过
            if os.path.exists(out) and os.path.getsize(out) == size:
                with lock:
                    done_size += size
                    print(f"[skip] {path} ({size/1e9:.2f}GB) 累计 {done_size/1e9:.1f}/{total/1e9:.1f}GB", flush=True)
                q.task_done()
                continue
            cmd = ["curl", "-sL", "-C", "-", "--retry", "10", "--retry-delay", "5",
                   "--connect-timeout", "30", "-o", out,
                   f"{BASE}/{path}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            got = os.path.getsize(out) if os.path.exists(out) else 0
            with lock:
                if got == size:
                    done_size += size
                    ok.append(path)
                    print(f"[ok] {path} ({size/1e9:.2f}GB) 累计 {done_size/1e9:.1f}/{total/1e9:.1f}GB", flush=True)
                else:
                    fail.append(path)
                    print(f"[FAIL] {path} got={got} want={size}", flush=True)
            q.task_done()

    q = Queue()
    for f in files:
        q.put(f)
    threads = [threading.Thread(target=worker) for _ in range(parallel)]
    for t in threads:
        t.start()
    q.join()
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()
    print(f"=== DONE ok={len(ok)} fail={len(fail)} 累计 {done_size/1e9:.1f}/{total/1e9:.1f}GB ===")
    if fail:
        print("failed:", fail)
        sys.exit(1)

if __name__ == "__main__":
    main()
