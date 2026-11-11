import datetime
import json
import os

import torch
from transformers import AutoModel, AutoTokenizer

P = "/mnt/raid/hliu553/models/Qwen3-Embedding-0.6B"
print("--- 目录 ---")
for f in sorted(os.listdir(P)):
    fp = os.path.join(P, f)
    ts = datetime.datetime.fromtimestamp(os.path.getmtime(fp))
    print("%12d %s %s" % (os.path.getsize(fp), ts, f))
cfg = json.load(open(os.path.join(P, "config.json")))
print("--- config 关键 ---")
for k in ("torch_dtype", "architectures", "model_type", "hidden_size", "num_hidden_layers"):
    print("  %s = %s" % (k, cfg.get(k)))

tok = AutoTokenizer.from_pretrained(P, trust_remote_code=True)
m = AutoModel.from_pretrained(P, trust_remote_code=True).eval()
print("model dtype =", next(m.parameters()).dtype)
enc = tok(["今天天气很好", "今天天气不错"], return_tensors="pt")
print("input_ids =", enc["input_ids"].tolist())
with torch.no_grad():
    out = m(**enc)
h = out.last_hidden_state
print("CPU last_hidden_state: nan =", bool(torch.isnan(h).any()),
      "inf =", bool(torch.isinf(h).any()),
      "mean =", float(h.mean()), "std =", float(h.std()))
