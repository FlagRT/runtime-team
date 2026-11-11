"""DeepFM-shape provider qualification; never creates a passing review."""
import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['qualification'], required=True)
args = parser.parse_args()
report = dict(status='ERROR', stage=args.stage, cases=[], contract={
    'operator': 'embedding', 'weight_shapes': [[3850,16],[3850,1]],
    'weight_dtype': 'torch.float32', 'indices_shape': [32,5],
    'indices_dtype': 'torch.int64', 'padding_idx': -1,
    'scale_grad_by_freq': False, 'sparse': False,
    'layout': 'contiguous', 'max_norm': None,
    'atol': 1e-5, 'rtol': 1e-4})


class Hits(logging.Handler):
    def __init__(self):
        super().__init__(); self.counts = Counter()
    def emit(self, record):
        if record.getMessage().startswith('GEMS EMBEDDING '):
            self.counts[record.getMessage()] += 1


try:
    import torch
    import torch.nn.functional as F
    import flag_gems as gems
    import runtime
    from environment import inspect_environment
    inspect_environment(args, report)
    hits = Hits()
    log = logging.getLogger('flag_gems.ops.embedding')
    log.setLevel(logging.DEBUG); log.addHandler(hits)
    tables = lambda: {op: torch._C._dispatch_dump_table('aten::'+op)
                      for op in ('embedding','embedding_backward','embedding_dense_backward')}
    for width in (16,1):
        generator = torch.Generator().manual_seed(20261110+width)
        weight = torch.randn(3850,width,generator=generator)
        grad = torch.randn(32,5,width,generator=generator)
        patterns = {
            'unique': torch.arange(160).reshape(32,5),
            'repeated': (torch.arange(160)%7).reshape(32,5),
            'all_same': torch.full((32,5),3,dtype=torch.int64),
            'boundary': (torch.arange(160)%2*3849).reshape(32,5)}
        for pattern, ids in patterns.items():
            cpu = weight.clone().requires_grad_()
            cpu_y = F.embedding(ids,cpu)
            cpu_y.backward(grad)
            x = weight.cuda().detach().requires_grad_()
            indices = ids.cuda(); dy = grad.cuda()
            native_y = F.embedding(indices,x)
            native_y.backward(dy); runtime.synchronize(0)
            native_fw = native_y.detach().cpu(); native_bw = x.grad.cpu()
            for a,b in ((native_fw,cpu_y.detach()),(native_bw,cpu.grad)):
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
            for repeat in range(3):
                before=tables(); hits.counts.clear()
                # Independent exact provider functions, no monkeypatching or autograd shortcut.
                y=gems.embedding(weight.cuda(),indices,-1,False,False)
                dx=gems.embedding_backward(dy,indices,3850,-1,False,False)
                runtime.synchronize(0)
                fw,bw=y.cpu(),dx.cpu()
                row=dict(width=width,pattern=pattern,repeat=repeat,status='pass',
                    hits=dict(hits.counts),restored=tables()==before,
                    forward_max_abs=(fw-native_fw).abs().max().item(),
                    backward_max_abs=(bw-native_bw).abs().max().item(),
                    cpu_backward_max_abs=(bw-cpu.grad).abs().max().item())
                try:
                    assert row['restored']
                    assert row['hits']=={'GEMS EMBEDDING FORWARD':1,'GEMS EMBEDDING BACKWARD':1}
                    for a,b in ((fw,native_fw),(bw,native_bw),(fw,cpu_y.detach()),(bw,cpu.grad)):
                        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
                except AssertionError:
                    row.update(status='fail',error=traceback.format_exc())
                report['cases'].append(row)
                print('CASE',json.dumps(row),flush=True)
    report['status']='PASS' if all(c['status']=='pass' for c in report['cases']) else 'FAIL'
except Exception:
    report['error']=traceback.format_exc(); print(report['error'],flush=True)
finally:
    Path('/results/result.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['status']=='PASS' else 1)
