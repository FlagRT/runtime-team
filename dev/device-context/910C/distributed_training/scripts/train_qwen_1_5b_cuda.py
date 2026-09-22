"""Qwen2.5-1.5B 两卡 DDP 训练 —— FlagCX flagcx backend（4090-1，NVIDIA adaptor）

与 910C 版（train_qwen_1_5b_npu.py）对应：device=cuda / backend=flagcx /
NCCL_IB_DISABLE=1（4090-1 无 IB，避免 NCCL IB 探测段错误）。
运行：
  NCCL_IB_DISABLE=1 torchrun --nproc_per_node=2 --master_port=29521 train_qwen_1_5b_cuda.py
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import time
import torch
import flagcx  # noqa: F401  注册 flagcx backend
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/home/data/hongbinliu/models/Qwen2.5-1.5B"
DATA_CACHE = "/home/data/hongbinliu/data"
OUT_DIR = "/home/data/hongbinliu/outputs"
MAX_LEN = 512
BATCH_SIZE = 1
EPOCHS = 1
LR = 5e-5
LOG_EVERY = 5
DEVICE = "cuda"


def main():
    dist.init_process_group(backend="flagcx")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dev = torch.device(DEVICE, local_rank)
    torch.cuda.set_device(local_rank)
    if rank == 0:
        print(f"[init] world_size={world_size}, local_rank={local_rank}, device={dev}", flush=True)
        print(f"[init] torch={torch.__version__}, cuda_count={torch.cuda.device_count()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True
    ).to(dev)
    model = DDP(model, device_ids=[local_rank])
    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"[model] params={n/1e6:.1f}M dtype=bf16", flush=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train", cache_dir=DATA_CACHE)

    def tok(ex):
        return tokenizer(ex["text"], add_special_tokens=False)["input_ids"]

    all_ids = []
    for ex in ds:
        if ex["text"].strip():
            all_ids.extend(tok(ex))
            all_ids.append(tokenizer.eos_token_id)
    chunks = [all_ids[i:i + MAX_LEN] for i in range(0, len(all_ids) - MAX_LEN, MAX_LEN)]
    if rank == 0:
        print(f"[data] tokens={len(all_ids)}, chunks={len(chunks)}", flush=True)

    class ChunkDS(Dataset):
        def __len__(self):
            return len(chunks)
        def __getitem__(self, i):
            ids = chunks[i]
            return {"input_ids": ids, "labels": list(ids), "attention_mask": [1] * len(ids)}

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = 0
        ii, ll, aa = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            p = maxlen - n
            ii.append(b["input_ids"] + [pad_id] * p)
            ll.append(b["labels"] + [-100] * p)
            aa.append([1] * n + [0] * p)
        return {
            "input_ids": torch.tensor(ii, dtype=torch.long),
            "labels": torch.tensor(ll, dtype=torch.long),
            "attention_mask": torch.tensor(aa, dtype=torch.long),
        }

    train_ds = ChunkDS()
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    t0 = time.time()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % LOG_EVERY == 0 and rank == 0:
                el = time.time() - t0
                tps = (step + 1) * BATCH_SIZE * world_size * MAX_LEN / max(el, 1e-6)
                print(f"[ep{epoch} s{step}] loss={loss.item():.4f} tok/s={tps:.0f}", flush=True)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        ckpt = os.path.join(OUT_DIR, "ckpt_final_flagcx_4090")
        model.module.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"[done] saved to {ckpt}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
