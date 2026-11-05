import sys, os, json, traceback, torch, torch_mlu
MODE = os.environ["PARITY_MODE"]
print("=== OP PARITY mode=%s ===" % MODE, flush=True)
torch.mlu.set_device(0)
print("flag_gems_in_modules_before", "flag_gems" in sys.modules, flush=True)
if MODE == "gems":
    import triton, flag_gems
    flag_gems.enable()
    print("ENABLE_OK triton=%s flag_gems=%s" % (triton.__version__, getattr(flag_gems, "__version__", "?")), flush=True)
print("device_count", torch.mlu.device_count(), flush=True)

SPECS = [
    ("gemm64",     lambda: (torch.randn(64,64,generator=torch.Generator().manual_seed(11)).to("mlu:0") @ torch.randn(64,64,generator=torch.Generator().manual_seed(12)).to("mlu:0"))),
    ("matmul",     lambda: (torch.randn(32,64,generator=torch.Generator().manual_seed(13)).to("mlu:0") @ torch.randn(64,16,generator=torch.Generator().manual_seed(14)).to("mlu:0"))),
    ("add",        lambda: (torch.randn(32,64,generator=torch.Generator().manual_seed(15)).to("mlu:0") + torch.randn(32,64,generator=torch.Generator().manual_seed(16)).to("mlu:0"))),
    ("mul",        lambda: (torch.randn(32,64,generator=torch.Generator().manual_seed(17)).to("mlu:0") * torch.randn(32,64,generator=torch.Generator().manual_seed(18)).to("mlu:0"))),
    ("silu",       lambda: torch.nn.functional.silu(torch.randn(32,64,generator=torch.Generator().manual_seed(19)).to("mlu:0"))),
    ("relu",       lambda: torch.relu(torch.randn(32,64,generator=torch.Generator().manual_seed(20)).to("mlu:0"))),
    ("softmax",    lambda: torch.softmax(torch.randn(32,64,generator=torch.Generator().manual_seed(21)).to("mlu:0"), dim=-1)),
    ("layernorm",  lambda: torch.nn.functional.layer_norm(torch.randn(32,64,generator=torch.Generator().manual_seed(22)).to("mlu:0"), (64,))),
    ("triu",       lambda: torch.triu(torch.randn(32,64,generator=torch.Generator().manual_seed(23)).to("mlu:0"))),
]
for name, fn in SPECS:
    try:
        out = fn()
        torch.mlu.synchronize()
        c = out.detach().cpu()
        row = {"op": name, "ok": True, "head": [round(float(v),6) for v in c.flatten()[:8].tolist()],
               "sum": round(float(c.sum()),6), "nan": int(torch.isnan(c).sum()), "inf": int(torch.isinf(c).sum()), "err": ""}
    except Exception as e:
        tb = traceback.format_exc().splitlines()
        row = {"op": name, "ok": False, "head": [], "sum": None, "nan": None, "inf": None,
               "err": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:180]),
               "last_frame": tb[-2].strip() if len(tb) >= 2 else ""}
    print("ROW:" + json.dumps(row, sort_keys=True), flush=True)
print("OP_PARITY_DONE", flush=True)
