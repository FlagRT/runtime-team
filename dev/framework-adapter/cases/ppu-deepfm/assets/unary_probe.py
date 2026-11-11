"""PPU qualification first; route tests consume separately reviewed evidence."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata as md
import json
import logging
import os
from pathlib import Path
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['qualification', 'routes'], required=True)
args = parser.parse_args()
report = {'status': 'ERROR', 'stage': args.stage, 'cases': []}


class Hits(logging.Handler):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def emit(self, record):
        message = record.getMessage()
        if message.startswith('GEMS '):
            self.counts[message] += 1


def tables(torch, op):
    return {s: torch._C._dispatch_dump_table('aten::' + s)
            for s in (op, op + '_backward')}


try:
    import torch
    import torch.nn.functional as F
    import triton
    import runtime
    import flag_gems as gems
    from flagrt_adapter.bridge import ElementwiseAdapter
    from flagrt_adapter.policy import Environment, OperatorPolicy
    from flagrt_adapter.records import load_records
    from flagrt_adapter.startup import package_content_identity

    runtime.use('ppu')
    assert runtime.device_count() == 1, 'Expected one mapped physical PPU'
    runtime.set_device(0)
    chip = torch.cuda.get_device_name(0)
    assert 'PPU' in chip, 'Refuse a non-PPU CUDA-compatible device'
    provider, detail = package_content_identity(gems.__file__)
    actual = Environment(chip=chip, torch=torch.__version__,
                         backend_version=os.environ['PPU_DRIVER_ID'],
                         gems_revision=provider, image_id=os.environ['CASE_IMAGE_ID'],
                         triton=triton.__version__, compiler_build=md.version('flagtree'),
                         backend='ppu', device_type='cuda')
    assert actual.complete()
    report.update(environment=asdict(actual), provider=detail,
                  versions={**{n: md.version(n) for n in ['flag-gems', 'flagtree', 'torch',
                            'flagrt-framework-adapter']}, 'triton-module': triton.__version__},
                  provider_route=os.environ.get('GEMS_VENDOR'),
                  runtime_version=runtime.__component_version__)
    print('ENVIRONMENT', report['environment'], flush=True)
    hits = Hits()
    for op in ('silu', 'sigmoid'):
        log = logging.getLogger('flag_gems.ops.' + op)
        log.setLevel(logging.DEBUG)
        log.addHandler(hits)
    records = load_records('/input/review/records.json') if args.stage == 'routes' else ()
    shape = (32,)
    for op, fn in [('sigmoid', torch.sigmoid)]:
        for dtype, atol, rtol in [(torch.float32, 1e-5, 1e-4)]:
            raw = torch.linspace(-5, 5, 32).reshape(shape).to(dtype)
            grad = torch.linspace(-1, 1, 32).reshape(shape).to(dtype)
            cpu = raw.float().clone().requires_grad_()
            ref = fn(cpu)
            ref.backward(grad.float())
            native = raw.to('cuda').detach().requires_grad_()
            native_out = fn(native)
            native_out.backward(grad.to('cuda'))
            runtime.synchronize(0)
            native_fw, native_bw = native_out.detach().float().cpu(), native.grad.float().cpu()
            torch.testing.assert_close(native_fw, ref.detach(), atol=atol, rtol=rtol)
            torch.testing.assert_close(native_bw, cpu.grad, atol=atol, rtol=rtol)
            modes = ['direct'] * 3 if args.stage == 'qualification' else ['off', 'on', 'unqualified']
            for index, mode in enumerate(modes):
                hits.counts.clear()
                before = tables(torch, op)
                x = raw.to('cuda').detach().requires_grad_()
                dy = grad.to('cuda')
                adapter = None
                if mode == 'direct':
                    # Qualification invokes the provider, not the admission bridge.
                    with gems.use_gems(include=[op, op + '_backward']):
                        y = getattr(torch.ops.aten, op).default(x)
                        saved = x if op == 'silu' else y
                        dx = getattr(torch.ops.aten, op + '_backward').default(dy, saved)
                        runtime.synchronize(0)
                else:
                    policy = OperatorPolicy(actual, (shape,), (str(dtype),),
                                            experimental=mode != 'off',
                                            capabilities=() if mode == 'unqualified' else records,
                                            operator=op)
                    adapter = ElementwiseAdapter({op: policy}, actual, runtime, gems)
                    with adapter:
                        y = fn(x)
                        y.backward(dy)
                    dx = x.grad
                    runtime.synchronize(0)
                assert tables(torch, op) == before, 'Dispatcher not restored'
                fw, bw = y.detach().float().cpu(), dx.detach().float().cpu()
                row = dict(operator=op, dtype=str(dtype), shape=list(shape), mode=mode,
                           repeat=index, atol=atol, rtol=rtol, restored=True,
                           hits=dict(hits.counts), status='pass',
                           forward_max_abs=(fw-native_fw).abs().max().item(),
                           backward_max_abs=(bw-native_bw).abs().max().item(),
                           cpu_forward_max_abs=(fw-ref.detach()).abs().max().item(),
                           cpu_backward_max_abs=(bw-cpu.grad).abs().max().item())
                try:
                    for a, b in [(fw, native_fw), (bw, native_bw), (fw, ref.detach()), (bw, cpu.grad)]:
                        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
                    expected_hits = {f'GEMS {op.upper()} FORWARD': 1,
                                     f'GEMS {op.upper()} BACKWARD': 1} if mode in ('direct', 'on') else {}
                    assert dict(hits.counts) == expected_hits, 'Actual provider entry count mismatch'
                    if adapter:
                        row['events'] = list(adapter.events)
                        expected_reason = {'off': 'experimental_disabled', 'unqualified': 'capability_unverified'}
                        if mode == 'on':
                            assert adapter.counts == Counter({('flaggems', 'forward', 'completed'): 1,
                                                              ('flaggems', 'backward', 'completed'): 1})
                        else:
                            assert adapter.counts == Counter({('native', 'forward', expected_reason[mode]): 1})
                        old_hits = dict(hits.counts)
                        outside = raw.to('cuda').detach().requires_grad_()
                        fn(outside).backward(dy)
                        runtime.synchronize(0)
                        assert dict(hits.counts) == old_hits, 'Replacement leaked after scope exit'
                        row['post_scope_native'] = True
                except AssertionError:
                    row['status'] = 'fail'
                    row['error'] = traceback.format_exc()
                report['cases'].append(row)
                print('CASE', json.dumps(row), flush=True)
    report['status'] = 'PASS' if all(c['status'] == 'pass' for c in report['cases']) else 'FAIL'
except Exception:
    report['error'] = traceback.format_exc()
    print(report['error'], flush=True)
finally:
    Path('/results/result.json').write_text(json.dumps(report, indent=2) + '\n')
raise SystemExit(0 if report['status'] == 'PASS' else 1)
