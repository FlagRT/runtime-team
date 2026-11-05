# 启用 FlagGems 后跑我方模型真实前向 —— 判定「不启用」是偏好还是当前不可用
import os, traceback, torch, torch_npu
torch.npu.set_device(0)
from transformers import AutoModel, AutoTokenizer
p = "/workspace/models/Qwen3-Embedding-0.6B"
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
m = AutoModel.from_pretrained(p, trust_remote_code=True).eval().to("npu:0")
import flag_gems
flag_gems.enable()
print("[gems] enabled", flush=True)
batch = tok("设备执行上下文层把各家芯片的分歧显式化。", return_tensors="pt", truncation=True, max_length=128).to("npu:0")
try:
    with torch.no_grad():
        out = m(**batch)
    torch.npu.synchronize()
    print("[gems-forward] OK  last_hidden[0,0]=%.6f" % float(out.last_hidden_state[0, 0, 0]), flush=True)
except Exception as exc:
    first = str(exc).splitlines()[0][:160]
    print("[gems-forward] FAILED  %s: %s" % (type(exc).__name__, first), flush=True)
    tb = traceback.format_exc().splitlines()
    for ln in tb[-14:]:
        print("   " + ln, flush=True)
