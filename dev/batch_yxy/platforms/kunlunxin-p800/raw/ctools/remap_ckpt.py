"""把 embedding 检查点的无前缀键名重映射为标准 Qwen3ForCausalLM 格式。

源: embed_tokens.weight / layers.N.* / norm.weight
目标: model.embed_tokens.weight / model.layers.N.* / model.norm.weight
输出写入容器内 /root/qwen3-gen/（物理落在 /data2 的容器可写层）。
"""

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("/work/models/qwen3-embed-0.6b-xpu")
DST = Path("/root/qwen3-gen")
DST.mkdir(parents=True, exist_ok=True)

sd = load_file(str(SRC / "model.safetensors"))
remapped = {}
for k, v in sd.items():
    if k.startswith("layers.") or k in ("embed_tokens.weight", "norm.weight"):
        remapped["model." + k] = v
    else:
        remapped[k] = v
save_file(remapped, str(DST / "model.safetensors"), metadata={"format": "pt"})
print(f"重映射 {len(remapped)} 个权重 -> {DST/'model.safetensors'}")

# 配置与 tokenizer 复制（config 保持 Qwen3ForCausalLM 架构）
cfg = json.loads((SRC / "config.json").read_text())
cfg["architectures"] = ["Qwen3ForCausalLM"]
cfg["tie_word_embeddings"] = True
(DST / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "generation_config.json"):
    shutil.copy(SRC / f, DST / f)
print("模型目录就绪:", sorted(p.name for p in DST.iterdir()))
