import sys, os, glob, json, traceback, torch, torch_mlu
MODE = os.environ["PARITY_MODE"]
print("=== FORWARD PARITY mode=%s ===" % MODE, flush=True)
torch.mlu.set_device(0)
if MODE == "gems":
    import triton, flag_gems
    flag_gems.enable()
    print("ENABLE_OK", flush=True)
from transformers import AutoModel, AutoTokenizer
snaps = glob.glob("/hf_cache/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*/")
p = snaps[0] if snaps else ""
print("model_path", p, flush=True)
try:
    tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
    m = AutoModel.from_pretrained(p, trust_remote_code=True).eval().to("mlu:0")
    batch = tok("设备执行上下文层把各家芯片的分歧显式化。", return_tensors="pt", truncation=True, max_length=128).to("mlu:0")
    with torch.no_grad():
        out = m(**batch)
    torch.mlu.synchronize()
    h = out.last_hidden_state[0, 0].detach().cpu()
    print("FORWARD_RESULT:" + json.dumps({"ok": True, "head": [round(float(v),6) for v in h[:8].tolist()],
        "sum": round(float(h.sum()),6), "nan": int(torch.isnan(h).sum()), "inf": int(torch.isinf(h).sum())}), flush=True)
except Exception as e:
    tb = traceback.format_exc().splitlines()
    print("FORWARD_RESULT:" + json.dumps({"ok": False,
        "err": "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:200]),
        "last_frame": tb[-2].strip() if len(tb) >= 2 else ""}), flush=True)
print("FORWARD_DONE", flush=True)
