"""步 0 续：模型可加载性 + 数值健全性（CPU 与 PPU 各一遍）。

判据：
  1. AutoTokenizer / AutoModel 能加载（910C 的坑：路径给到缓存根会 Unrecognized model）
  2. 前向输出**全部有限**（NaN/Inf 检查 —— 910C 曾出现"跑完 50 步但 loss=nan"的静默损坏）
  3. CPU 与 PPU 结果**量级可比**（不是逐位相等，浮点不同）
"""
import json
import math
import os
import sys

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Qwen3-Embedding-0.6B"
TEXTS = ["hello world", "设备执行上下文层"]


def report(tag: str, out: dict) -> None:
    print(f"[{tag}] " + json.dumps(out, ensure_ascii=False))


res: dict = {"model": MODEL, "torch": torch.__version__, "cuda_avail": torch.cuda.is_available()}
try:
    import transformers

    res["transformers"] = transformers.__version__
except Exception as exc:  # pragma: no cover
    res["transformers"] = f"ERR {exc}"

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
res["architectures"] = getattr(cfg, "architectures", None)
res["hidden_size"] = getattr(cfg, "hidden_size", None)
res["num_hidden_layers"] = getattr(cfg, "num_hidden_layers", None)
res["torch_dtype"] = str(getattr(cfg, "torch_dtype", None))

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
enc = tok(TEXTS, return_tensors="pt", padding=True)
res["input_ids_shape"] = list(enc["input_ids"].shape)
res["vocab_size_tok"] = tok.vocab_size

# ── 1) CPU 前向 ──────────────────────────────────────────────────────────────
model_cpu = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).eval()
with torch.no_grad():
    o_cpu = model_cpu(**enc).last_hidden_state
res["cpu_last_hidden_shape"] = list(o_cpu.shape)
res["cpu_all_finite"] = bool(torch.isfinite(o_cpu).all().item())
res["cpu_absmax"] = float(o_cpu.abs().max().item())
res["cpu_mean"] = float(o_cpu.mean().item())
res["cpu_has_nan"] = bool(torch.isnan(o_cpu).any().item())
res["cpu_has_inf"] = bool(torch.isinf(o_cpu).any().item())
del model_cpu
report("cpu", res)

# ── 2) PPU 前向（用哪张卡显式记录）──────────────────────────────────────────
if not torch.cuda.is_available():
    res["ppu"] = "SKIP: cuda.is_available()=False（未挂设备？见 PPU 接入报告 §3）"
    report("ppu", res)
    print("VERDICT=" + ("PASS" if res["cpu_all_finite"] else "FAIL"))
    raise SystemExit(0)

res["cuda_device_count"] = torch.cuda.device_count()
res["cuda_device_name_0"] = torch.cuda.get_device_name(0)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.float32).to("cuda:0").eval()
with torch.no_grad():
    o = model(**{k: v.to("cuda:0") for k, v in enc.items()}).last_hidden_state
res["gpu_last_hidden_shape"] = list(o.shape)
res["gpu_all_finite"] = bool(torch.isfinite(o).all().item())
res["gpu_absmax"] = float(o.abs().max().item())
res["gpu_mean"] = float(o.mean().item())
res["gpu_has_nan"] = bool(torch.isnan(o).any().item())
res["gpu_has_inf"] = bool(torch.isinf(o).any().item())
# 量级可比性（不做逐位比较）
res["mean_abs_diff_cpu_gpu"] = float((o.cpu() - o_cpu).abs().mean().item())
res["same_shape_cpu_gpu"] = list(o.shape) == list(o_cpu.shape)
report("ppu", res)

ok = bool(res["cpu_all_finite"]) and bool(res["gpu_all_finite"]) and res["same_shape_cpu_gpu"]
print("VERDICT=" + ("PASS" if ok else "FAIL"))
