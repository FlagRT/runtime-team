"""One full native training step on each pinned PPU model; no target replacement."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

from trace_ops import Inventory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def run(a, report):
    import torch
    import torch.nn.functional as F
    from environment import inspect_environment
    inspect_environment(SimpleNamespace(runtime_root=a.runtime_root), report)
    import runtime
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(a.model)
    seed = {'deepfm': 20261023, 'qwen': 1234, 'onerec': 1014}[a.kind]
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if a.kind == 'deepfm':
        from deepfm_probe import load_model, make_batch, MODEL_HASHES
        model, config = load_model(path)
        files = dict(MODEL_HASHES)
        inputs, labels = make_batch(config, 32, 0)
        model = model.to('cuda:0').train()
        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-8, foreach=False)
        settings = dict(batch=32, dtype='float32', seed=seed, data_seed=29000,
                        optimizer='Adam', lr=1e-3, dropout=config.dropout)
        torch.manual_seed(23000); torch.cuda.manual_seed_all(23000)
    else:
        from transformers import AutoModelForCausalLM
        from onerec_probe import WEIGHTS, verify_architecture
        weights = WEIGHTS if a.kind == 'onerec' else {
            'model.safetensors': '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd'}
        files = {name: sha(path / name) for name in weights}
        if files != weights: raise ValueError('Pretrained weight identity mismatch')
        files['config.json'] = sha(path / 'config.json')
        if a.kind == 'onerec': verify_architecture(json.loads((path / 'config.json').read_text()))
        dtype = torch.bfloat16 if a.kind == 'onerec' else torch.float32
        model, loading = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, dtype=dtype, attn_implementation='eager', output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Incomplete checkpoint')
        model = model.to('cuda:0').train(); model.config.use_cache = False
        if a.kind == 'onerec':
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, foreach=False)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=.01, foreach=False)
        sequence = 16 if a.kind == 'onerec' else 32
        data_seed = 1014 if a.kind == 'onerec' else 999
        inputs = torch.randint(0, model.config.vocab_size, (1, sequence),
                               generator=torch.Generator().manual_seed(data_seed)).to('cuda:0')
        labels = inputs
        settings = dict(batch=1, sequence=sequence, dtype=str(dtype), seed=seed,
                        data_seed=data_seed, attention='eager', gradient_checkpointing=a.kind=='onerec')
    expected = {'deepfm': (421452, 0), 'qwen': (595776512, 310), 'onerec': (8391013376, 399)}
    count = sum(p.numel() for p in model.parameters())
    if count != expected[a.kind][0]: raise ValueError('Wrong model size')
    source = Path(inspect.getfile(type(model)))
    report.update(model=a.kind, model_files=files, settings=settings, parameter_count=count,
                  model_class=type(model).__name__, model_source_sha256=sha(source),
                  model_source_file=str(source), stage='forward', optimizer_steps=0)
    inventory = Inventory(); inventory.attach(model)
    def save():
        report['inventory'] = inventory.report()
        a.out.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        optimizer.zero_grad(set_to_none=True)
        with inventory.scope('forward'):
            loss = F.binary_cross_entropy(model(inputs), labels) if a.kind=='deepfm' else model(
                input_ids=inputs, labels=labels, use_cache=False).loss
        if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite loss')
        report.update(loss=float(loss.detach()), stage='backward'); save()
        with inventory.scope('backward'): loss.backward()
        runtime.synchronize(0)
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError('Missing or nonfinite gradient')
        report['stage']='optimizer'; save()
        with inventory.scope('optimizer'):
            if a.kind=='qwen': torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=False)
            optimizer.step()
        runtime.synchronize(0)
        report.update(status='PASS', stage='complete', optimizer_steps=1)
        assert not inventory.stack
    finally:
        inventory.detach(); save()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=('deepfm','qwen','onerec'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--runtime-root',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=dict(status='ERROR', scope='native training eager operator inventory, one step',
                source_sha256={n:sha(Path(__file__).parent/n) for n in ('probe.py','trace_ops.py')})
    try: run(a,report)
    except Exception:report.update(status='ERROR',error=traceback.format_exc());traceback.print_exc()
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(report['status'],a.kind,flush=True)
    sys.exit(0 if report['status']=='PASS' else 2)
