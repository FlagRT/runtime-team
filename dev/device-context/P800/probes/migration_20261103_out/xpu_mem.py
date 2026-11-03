# 解析 xpu-smi，逐卡输出已用显存（MiB）
import re, subprocess
out = subprocess.run(["xpu-smi"], capture_output=True, text=True).stdout
cur, rows = None, []
for line in out.splitlines():
    f = [x.strip() for x in line.split("|")]
    if len(f) < 4:
        continue
    if "P800 OAM" in f[1]:
        m = re.match(r"(\d+)", f[1]); cur = m.group(1) if m else cur
        continue
    mm = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", f[2])
    if mm and cur is not None:
        rows.append((cur, int(mm.group(1)), int(mm.group(2))))
seen = set()
for c, u, t in rows:
    if c in seen: continue
    seen.add(c)
    print("card%-2s used=%6d MiB / %d MiB  %.1f%%" % (c, u, t, u/t*100 if t else 0))
