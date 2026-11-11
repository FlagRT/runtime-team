import time
import torch
import torch_br  # noqa: F401

dev = torch.supa
dev.set_device(0)
torch.zeros(1, device="supa")


def run(n, nb):
    hosts = [torch.randn(n, n).pin_memory() for _ in range(2)]
    buf = [torch.zeros(n, n, device="supa") for _ in range(2)]
    ev_h = [dev.Event() for _ in range(2)]
    ev_c = [dev.Event() for _ in range(2)]
    s_h2d, s_mm, s_d2h = dev.Stream(), dev.Stream(), dev.Stream()

    def pipe():
        oc = None
        for i in range(nb):
            b = i % 2
            with dev.stream(s_h2d):
                buf[b].copy_(hosts[b], non_blocking=True)
            ev_h[b].record(s_h2d)
            s_mm.wait_event(ev_h[b])
            with dev.stream(s_mm):
                out = (buf[b] @ buf[b]).sum()
            ev_c[b].record(s_mm)
            s_d2h.wait_event(ev_c[b])
            with dev.stream(s_d2h):
                oc = out.cpu()
        dev.synchronize()
        return oc

    def serial():
        r = None
        for i in range(nb):
            d = hosts[i % 2].to("supa")
            r = (d @ d).sum().cpu()
        dev.synchronize()
        return r

    pipe()
    serial()  # 预热（含一次性编译）
    t0 = time.time()
    pipe()
    pt = time.time() - t0
    t0 = time.time()
    serial()
    stt = time.time() - t0
    print("  n=%4d batches=%2d | pipe=%.4fs serial=%.4fs 重叠率=%+.1f%%"
          % (n, nb, pt, stt, (stt - pt) / stt * 100))


print("=== i6 口径随工作量变化（预热后计时）===")
for n, nb in ((512, 4), (1024, 8), (2048, 8), (2048, 16), (4096, 8)):
    try:
        run(n, nb)
    except Exception as exc:
        print("  n=%d batches=%d FAIL %s: %s" % (n, nb, type(exc).__name__, str(exc)[:80]))
print("=== done ===")
