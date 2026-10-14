"""Small real-data embedding fine-tuning test, NOT full NFCorpus evaluation.

Reads the BEIR archive without redistributing text. Official train/dev query
splits, deterministic pair selection, in-batch unjudged negatives. FP32 full
parameter updates, optional isolated-process SiLU forward/backward registration.
"""
import argparse
from contextlib import nullcontext
import hashlib
import io
import csv
import json
import logging
from pathlib import Path
import sys
import traceback
import zipfile

from test_silu_training_npu import Calls


def load_data(path):
    with zipfile.ZipFile(path) as z:
        def lines(name):
            return io.TextIOWrapper(z.open("nfcorpus/" + name), encoding="utf-8")
        corpus = {r["_id"]: r for r in map(json.loads, lines("corpus.jsonl"))}
        queries = {r["_id"]: r["text"] for r in map(json.loads, lines("queries.jsonl"))}
        rels = {}
        for split in ("train", "dev"):
            rels[split] = {}
            for r in csv.DictReader(lines("qrels/" + split + ".tsv"), delimiter="\t"):
                if int(r["score"]) > 0:
                    rels[split].setdefault(r["query-id"], {})[r["corpus-id"]] = int(r["score"])
    assert not (set(rels["train"]) & set(rels["dev"]))
    dev_texts = {queries[q].strip().lower() for q in rels["dev"]}
    pairs, used_docs, used_queries = [], set(), set()
    for q in sorted(rels["train"]):
        if queries[q].strip().lower() in dev_texts:
            continue
        choices = sorted(rels["train"][q], key=lambda d: (-rels["train"][q][d], d))
        for d in choices:
            if d not in corpus or d in used_docs:
                continue
            # Within each pair-of-pairs, exclude all known cross positives.
            if len(pairs) % 2 and (d in rels["train"][pairs[-1][0]] or pairs[-1][1] in rels["train"][q]):
                continue
            pairs.append((q, d)); used_docs.add(d); used_queries.add(q)
            break
        if len(pairs) == 20:
            break
    assert len(pairs) == 20
    dev = []
    candidates = []
    for q in sorted(rels["dev"]):
        options = [d for d in sorted(rels["dev"][q], key=lambda d: (-rels["dev"][q][d], d))
                   if d not in used_docs and d in corpus]
        if options:
            dev.append(q)
            if options[0] not in candidates:
                candidates.append(options[0])
        if len(dev) == 8:
            break
    positive_union = {d for q in dev for d in rels["dev"][q]}
    for d in sorted(corpus):
        if d not in positive_union and d not in used_docs:
            candidates.append(d)
        if len(candidates) == 32:
            break
    assert len(dev) == 8 and len(candidates) == 32
    return corpus, queries, rels, pairs, dev, candidates


def run(args, result):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, args.runtime_root)
    import runtime
    runtime.use("ascend"); runtime.device_count(); runtime.set_device(0)
    from transformers import AutoModel, AutoTokenizer
    import flag_gems
    corpus, queries, rels, pairs, dev, candidates = load_data(args.data)
    result["data"] = {"source": "BEIR NFCorpus official archive", "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "train_pairs": pairs, "dev_queries": dev, "dev_candidate_documents": candidates,
        "negative_semantics": "unjudged, not confirmed irrelevant; known cross positives excluded"}
    torch.manual_seed(1234)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").to("npu:0")
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    calls = Calls()
    logging.getLogger("flag_gems").setLevel(logging.DEBUG)
    logging.getLogger("flag_gems").addHandler(calls)
    before = {op: torch._C._dispatch_dump_table("aten::"+op) for op in ("silu", "silu_backward")}
    def query(q):
        return "Instruct: Given a question, retrieve relevant documents that answer the question\nQuery: " + queries[q]
    def doc(d):
        r = corpus[d]
        return (r.get("title", "") + " " + r["text"]).strip()
    def encode(texts):
        batch = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to("npu:0")
        h = model(**batch, use_cache=False).last_hidden_state
        last = (torch.arange(batch["attention_mask"].shape[1], device="npu:0") * batch["attention_mask"]).max(dim=1).values
        return F.normalize(h[torch.arange(len(texts), device="npu:0"), last].float(), dim=-1)
    def evaluate():
        model.eval()
        with torch.no_grad():
            q = torch.cat([encode([query(k)]) for k in dev])
            d = torch.cat([encode([doc(k) for k in candidates[i:i+2]]) for i in range(0, len(candidates), 2)])
            scores = (q @ d.T).cpu()
        assert bool(torch.isfinite(scores).all())
        ranks = []
        for i, qid in enumerate(dev):
            ordered = scores[i].argsort(descending=True).tolist()
            ranks.append(next(rank+1 for rank, col in enumerate(ordered) if candidates[col] in rels["dev"][qid]))
        return {"candidate_pool_size": len(candidates), "queries": len(dev), "relevant_first_ranks": ranks,
                "hit_at_1": sum(r == 1 for r in ranks)/len(ranks),
                "mrr": sum(1/r for r in ranks)/len(ranks), "scores": scores.tolist()}
    result["before"] = evaluate()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    sampled = next(p for n, p in model.named_parameters() if n.endswith("layers.0.mlp.down_proj.weight"))
    initial = sampled.detach().cpu().clone()
    result["steps"] = []
    for step in range(10):
        model.train()
        torch.manual_seed(1000+step)
        batch_pairs = pairs[2*step:2*step+2]
        old_counts = calls.counts.copy()
        opt.zero_grad(set_to_none=True)
        context = flag_gems.use_gems(include=["silu", "silu_backward"]) if args.mode == "gems" else nullcontext()
        with context:
            vectors = encode([query(q) for q, _ in batch_pairs] + [doc(d) for _, d in batch_pairs])
            scores = vectors[:2] @ vectors[2:].T / 0.05
            loss = F.cross_entropy(scores, torch.arange(2, device="npu:0"))
            assert bool(torch.isfinite(loss))
            loss.backward()
            runtime.synchronize(0)
        delta = calls.counts - old_counts
        if args.mode == "gems":
            assert delta["GEMS SILU FORWARD"] > 0 and delta["GEMS SILU BACKWARD"] > 0
        else:
            assert not delta
        assert all(torch._C._dispatch_dump_table("aten::"+op) == table for op, table in before.items())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        count = sum(p.grad is not None for p in model.parameters())
        row = {"step": step, "query_ids": [q for q, _ in batch_pairs], "loss": float(loss.detach()),
               "gradient_norm": float(norm), "gradient_tensors": count, "gems_calls": dict(delta)}
        opt.step(); runtime.synchronize(0)
        result["steps"].append(row)
        print(json.dumps(row), flush=True)
    result["sampled_parameter_max_update"] = float((sampled.detach().cpu()-initial).abs().max())
    assert result["sampled_parameter_max_update"] > 0
    result["after"] = evaluate()  # Both checkpoints evaluated with native ops.
    result["dispatcher_restored"] = all(torch._C._dispatch_dump_table("aten::"+op) == table for op, table in before.items())
    if args.checkpoint:
        if Path(args.checkpoint).exists():
            raise FileExistsError(args.checkpoint)
        model.save_pretrained(args.checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(args.checkpoint)
        result["checkpoint"] = args.checkpoint
    result["status"] = "pass"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("model", "runtime-root", "data", "out"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--mode", choices=("native", "gems"), required=True)
    p.add_argument("--checkpoint")
    args = p.parse_args()
    if Path(args.out).exists() or (args.checkpoint and Path(args.checkpoint).exists()):
        p.error("Output and checkpoint must not already exist")
    result = {"status": "fail", "mode": args.mode, "scope": "20-pair real-data miniature embedding training",
              "config": {"dtype": "float32", "steps": 10, "pair_batch": 2, "max_tokens": 128,
                         "lr": 1e-5, "temperature": 0.05, "optimizer": "AdamW", "seed": 1234}}
    try:
        run(args, result)
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
    finally:
        with open(args.out, "x") as f:
            json.dump(result, f, indent=2)
    sys.exit(0 if result["status"] == "pass" else 1)
