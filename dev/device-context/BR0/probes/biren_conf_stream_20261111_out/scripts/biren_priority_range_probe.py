import torch, torch_br
dev = torch.supa
print("=== Stream.priority_range 定性 ===")
try:
    r = dev.Stream.priority_range()
    print("priority_range() =", r, "| type =", type(r).__name__)
    if isinstance(r, (tuple, list)) and len(r) >= 2:
        least, greatest = r[0], r[1]
        print("least=%r greatest=%r  => 档位数=%r" % (least, greatest, (greatest - least + 1) if isinstance(least, int) and isinstance(greatest, int) else "n/a"))
except Exception as exc:
    print("priority_range() 失败:", type(exc).__name__, str(exc).splitlines()[0][:120])

print("\n=== 合法档内设置-回读一致性 ===")
for prio in (0, -1, -2, 1):
    try:
        s = dev.Stream(priority=prio)
        print("  set %3d -> readback %r" % (prio, s.priority))
    except Exception as exc:
        print("  set %3d -> FAIL %s: %s" % (prio, type(exc).__name__, str(exc).splitlines()[0][:70]))
print("=== done ===")
