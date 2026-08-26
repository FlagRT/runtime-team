"""
Qwen2.5-1.5B 双卡 DDP 训练 —— torch_npu + FlagCX flagcx backend（910C，A 线）
==========================================================================
目标：验证官方 dev-1.0 HCCL 适配 + 四层根因修复在 torch_npu 环境下的
      910C 两卡训练闭环（本机同构，不依赖 torch_fl）
要点：
  - import torch_npu 注册 npu 设备（PrivateUse1）
  - dist backend="flagcx"（FlagCX torch 插件，HCCL adaptor）
  - dtype bf16；模型 /data_lib/models/Qwen2.5-1.5B
运行（宿主侧）：
  docker exec flagos-device-context-a-910c bash -lc "cd /workspace/dev/device-context/benchmarks &&   torchrun --nproc_per_node=2 --master_port=29521 train_qwen_1_5b_npu.py"
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import time
import torch
import torch_npu  # noqa: F401  # 注册 npu 设备后端
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_PATH = "/workspace/models/Qwen2.5-1.5B"
DATA_CACHE = "/workspace/data"
OUT_DIR = "/workspace/outputs"
MAX_LEN = 512
BATCH_SIZE = 1
EPOCHS = 1
LR = 5e-5
LOG_EVERY = 5
DEVICE = "npu"


def main():
    # ---- 分布式初始化（A 线：FlagCX flagcx backend = HCCL adaptor）----
    BACKEND = os.environ.get("BACKEND", "flagcx")
    dist.init_process_group(backend=BACKEND)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- 设备绑定（每个进程绑定自己的 NPU）----
    dev = torch.device(DEVICE, local_rank)
    torch.npu.set_device(local_rank)
    if rank == 0:
        print(f"[init] world_size={world_size}, local_rank={local_rank}, device={dev}", flush=True)
        print(f"[init] torch={torch.__version__}, npu_count={torch_npu.npu.device_count()}", flush=True)

    # ---- 模型 ----
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

    # ---- 数据：wikitext-2-raw-v1（缓存已就位），flatten 切块 ----
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

    # ---- 训练循环 ----
    t0 = time.time()
    # ---- 可选 profiler（PROFILE=1 启用）：统计 allreduce 调用次数与耗时 ----
    use_prof = os.environ.get("PROFILE") == "1"
    prof = None
    if use_prof:
        from torch.profiler import profile, ProfilerActivity, schedule
        prof = profile(activities=[ProfilerActivity.CPU],
                       schedule=schedule(wait=10, warmup=5, active=20, repeat=0))
        prof.start()
    for epoch in range(EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        for step, batch in enumerate(loader):
            max_steps = int(os.environ.get("MAX_STEPS", "0"))
            if max_steps and step >= max_steps:
                break
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
            if use_prof:
                prof.step()

    if use_prof:
        prof.stop()
        if rank == 0:
            keys = prof.key_averages()
            ars = [e for e in keys if "allreduce" in e.key.lower() or "all_reduce" in e.key.lower()]
            n_ar = len(ars)
            tot_ms = sum(e.self_cpu_time_total for e in ars) / 1000.0
            n_all = sum(1 for e in keys if "allreduce" in e.key.lower())
            n_calls = sum(e.count for e in ars)
            avg_ms = tot_ms / n_calls if n_calls else 0.0
            print(f"[profiler] backend={BACKEND} allreduce_events={n_ar} calls={n_calls} total_self_cpu_ms={tot_ms:.1f} avg_self_cpu_ms={avg_ms:.2f}", flush=True)
            import os as _os
            _os.makedirs("/workspace/logs", exist_ok=True)
            prof.export_chrome_trace(f"/workspace/logs/trace_{BACKEND}.json")

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        ckpt = os.path.join(OUT_DIR, "ckpt_final_npu")
        model.module.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"[done] saved to {ckpt}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
