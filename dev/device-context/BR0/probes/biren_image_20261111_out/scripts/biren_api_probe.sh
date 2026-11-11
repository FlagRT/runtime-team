#!/usr/bin/env bash
set -u
C=dc-biren-hliu553
echo "=== SELF_CHECK biren api probe 2026-11-11 ==="
cat > /tmp/api.py <<'PY'
import torch, torch_br, inspect
s = torch.supa
names = [n for n in dir(s) if not n.startswith("_")]
print("supa attrs(", len(names), "):", ", ".join(sorted(names)))
print()
for fn in ["Stream","Event","current_stream","default_stream","stream","synchronize","set_device",
           "device_count","current_device","get_device_name","mem_get_info","memory_allocated",
           "get_device_properties","empty_cache","device"]:
    print("%-22s %s" % (fn, hasattr(s, fn)))
try:
    st = s.Stream()
    print("\nStream() ok:", st, "has synchronize:", hasattr(st,"synchronize"), "has query:", hasattr(st,"query"))
    print("Stream(priority=0):", end=" ")
    try: print(s.Stream(priority=0))
    except Exception as e: print("FAIL", type(e).__name__, str(e)[:60])
except Exception as e:
    print("Stream() FAIL", type(e).__name__, str(e)[:80])
try:
    ev = s.Event(enable_timing=True)
    print("Event() ok:", ev, "| query(未record):", ev.query(), "| has elapsed_time:", hasattr(ev,"elapsed_time"))
    ev.record(); print("after record, query:", ev.query()); s.synchronize(); print("after sync, query:", ev.query())
except Exception as e:
    print("Event FAIL", type(e).__name__, str(e)[:80])
print("mem_get_info(0):", s.mem_get_info(0) if hasattr(s,"mem_get_info") else "n/a")
print("get_device_properties(0) type:", type(s.get_device_properties(0)).__name__ if hasattr(s,"get_device_properties") else "n/a")
PY
docker cp /tmp/api.py "$C":/tmp/api.py >/dev/null 2>&1
docker exec "$C" bash -c '. /usr/local/birensupa/br_container_tools/brsw_set_env.sh --not-set-env-if-already-set >/dev/null 2>&1; python3 /tmp/api.py' 2>&1 | grep -viE '^\s*warn|UserWarning' | head -22
docker exec "$C" rm -f /tmp/api.py; rm -f /tmp/api.py
