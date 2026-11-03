"""Bounded P800 unary qualification/routing. No kernel edits or automatic approval."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import subprocess
import traceback

IMAGE = 'sha256:cac5ddae5f62b356622858dd4f0ea0f82d489cf0cb71e7230f0f97797c25605a'
GEMS = '73c5aff1e304c293ab525ecb03e9449644496fbb'
SHAPES = ((32, 64), (1, 32, 3072))
OP_SHAPES = {'silu': SHAPES, 'sigmoid': ((16,), (32,)), 'tanh': ((16,), (32,))}


def selected_shapes(operator, profile):
    if profile == 'default': return OP_SHAPES[operator]
    if profile == 'onerec16' and operator == 'silu': return ((1, 16, 12288),)
    raise ValueError('Unknown or incompatible shape profile')

BAKED_ARTIFACTS = {
    '?? output.tar.gz', '?? output/.scm/module_info', '?? output/version.txt',
    '?? output/triton-xpu-ubuntu_2004_x86_64/triton-3.0.0+03c4c9be-cp310-cp310-linux_x86_64.whl',
}


def baked_inventory_acceptable(inventory):
    return (inventory['head'] == GEMS and not inventory['diff_stat']
            and not inventory['tracked_diff']
            and set(inventory['status'].splitlines()) == BAKED_ARTIFACTS)


class Hits(logging.Handler):
    def __init__(self, operator='silu'):
        super().__init__()
        self.counts = Counter()
        self.operator = operator

    def emit(self, record):
        text = record.getMessage()
        if self.operator in text.lower():
            self.counts[text] += 1


def backward_argument(operator, source, output):
    """ATen sigmoid/tanh backward consume saved OUTPUT, not the input."""
    if operator not in OP_SHAPES:
        raise ValueError('Unsupported unary operator')
    return source if operator == 'silu' else output


def versions():
    found = {}
    for name in ('torch', 'torch_xmlir', 'torch-xray', 'xtorch_ops', 'flagtree', 'flag_gems'):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def run(args, report):
    import torch
    import torch.nn.functional as F
    import triton
    import flagrt_adapter
    from flagrt_adapter import Environment, load_records
    from flagrt_adapter.startup import package_identity, package_content_identity, prepare_policies
    from flagrt_adapter.bridge import ElementwiseAdapter
    if Path('/results/installed') not in Path(flagrt_adapter.__file__).resolve().parents:
        raise RuntimeError('Adapter must come from the offline-installed wheel')
    sys.path.insert(0, args.runtime_root)
    import runtime
    # Follow image guidance: do not enable the known-hanging KL3 event option.
    if os.environ.get('XPU_EVENT_KL3_ENABLE') or 'gemm_compat' in sys.modules:
        raise RuntimeError('Unexpected KL3/GEMM opt-in; this case qualifies the unmodified base')
    runtime.use('kunlun')
    runtime.set_device(0)
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Requires exactly one mapped P800, not all host devices')
    chip = torch.cuda.get_device_name(0)
    xmlir_spec = importlib.util.find_spec('torch_xmlir')
    report['device_detection'] = dict(reported_name=chip, torch_xmlir_path=xmlir_spec.origin if xmlir_spec else None,
                                      kunlun_driver_present=Path('/proc/kunlun').exists())
    # XPytorch may expose a CUDA-compatible marketing name. Require both the
    # vendor module and driver, plus the externally pinned P800-only image.
    if xmlir_spec is None or not Path('/proc/kunlun').exists():
        raise RuntimeError('No joint vendor-module/driver evidence: '+str(report['device_detection']))
    provider_file = importlib.util.find_spec('flag_gems').origin
    provider_root = Path(provider_file).resolve().parents[2]
    git = ['git','-c','safe.directory='+str(provider_root),'-C',str(provider_root)]
    report['provider_inventory'] = {
        'path': provider_file,
        'head': subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip(),
        'status': subprocess.check_output(git+['status','--porcelain','--untracked-files=all'],text=True),
        'diff_stat': subprocess.check_output(git+['diff','HEAD','--stat'],text=True),
        'tracked_diff': subprocess.check_output(git+['diff','HEAD','--','src/flag_gems'],text=True),
    }
    # Exact audited image-local build artifacts only. Do not silently accept a
    # modified checkout or call these bytes the clean upstream Git revision.
    if report['provider_inventory']['status']:
        if str(provider_root) != '/env/FlagGems' or not baked_inventory_acceptable(report['provider_inventory']):
            raise RuntimeError('Unreviewed baked provider differences')
        revision, identity = package_content_identity(provider_file)
        identity['source_head'] = GEMS
        identity['audited_inert_artifacts'] = sorted(BAKED_ARTIFACTS)
    else:
        revision, identity = package_identity(provider_file)
        if revision != GEMS:
            raise RuntimeError('Unexpected provider identity: ' + revision)
    installed = versions()
    backend_version = ';'.join(name+'=='+installed[name] for name in ('torch_xmlir', 'xtorch_ops')
                               if installed[name])
    if os.environ.get('FLAGRT_VERIFIED_CHIP') != 'P800':
        raise RuntimeError('Host runner must verify the physical chip via xpu-smi')
    actual = Environment('P800', torch.__version__, backend_version, revision, IMAGE,
                         triton.__version__, 'flagtree=='+str(installed['flagtree']),
                         backend='kunlunxin', device_type='cuda')
    if not actual.complete() or installed['flagtree'] != '0.6.1+xpu3.6':
        raise RuntimeError('Incomplete or unexpected compiler/backend identity')
    report.update(environment=asdict(actual), packages=installed, provider_identity=identity,
                  device_count=torch.cuda.device_count(), adapter_version=flagrt_adapter.__version__,
                  paths=dict(torch=torch.__file__, triton=triton.__file__, provider=provider_file,
                             runtime=runtime.__file__, adapter=flagrt_adapter.__file__),
                  gemm_compat_enabled=False, cases=[])
    def compare(a, b, tol):
        a, b = a.detach().float().cpu(), b.detach().float().cpu()
        torch.testing.assert_close(a, b, rtol=tol, atol=tol)
        return float((a-b).abs().max())
    # Native path is a separate prerequisite, before any registration.
    cpu = torch.linspace(-8, 8, 2048).reshape(32,64).requires_grad_()
    grad_cpu = torch.cos(cpu.detach()).contiguous()
    x = cpu.detach().to('cuda:0').requires_grad_()
    y = F.silu(x)
    dx, = torch.autograd.grad(y, x, grad_cpu.to('cuda:0'))
    cy = F.silu(cpu); cg, = torch.autograd.grad(cy, cpu, grad_cpu)
    runtime.synchronize(0)
    report['native_smoke'] = dict(forward_max_abs=compare(y,cy,1e-5),
                                  backward_max_abs=compare(dx,cg,1e-5), status='pass')
    if args.stage == 'inspect':
        report['status'] = 'pass'
        return
    operator = getattr(args, 'operator', 'silu')
    forward = F.silu if operator == 'silu' else getattr(torch, operator)
    ops = (operator, operator+'_backward')
    report['operator'] = operator
    records = load_records(args.records) if args.records else ()
    gems = None
    if args.stage == 'qualification' or args.mode == 'on':
        import flag_gems as gems
    hits = Hits(operator); logger = logging.getLogger('flag_gems')
    logger.setLevel(logging.DEBUG); logger.addHandler(hits)
    try:
        for shape in selected_shapes(operator, getattr(args, 'shape_profile', 'default')):
            for dtype, tol in ((torch.float32,1e-5),(torch.float16,5e-3),(torch.bfloat16,3e-2)):
                torch.manual_seed(20261029)
                # Generate on CPU to hold input constant across repeats/backends.
                source = torch.randn(shape).to(dtype)
                upstream = torch.randn(shape).to(dtype)
                x = source.to('cuda:0').requires_grad_()
                dy = upstream.to('cuda:0').contiguous()
                expected = forward(x)
                expected_grad, = torch.autograd.grad(expected, x, dy)
                before = {op:torch._C._dispatch_dump_table('aten::'+op)
                          for op in ops}
                item = dict(operator=operator, shape=list(shape), dtype=str(dtype), mode=args.mode,
                            status='running', tolerance=tol, repeats=[])
                report['cases'].append(item)
                for repeat in range(3):
                    previous_hits = hits.counts.copy()
                    adapter = None
                    try:
                        if args.stage == 'qualification':
                            with torch.no_grad(), gems.use_gems(include=list(ops)):
                                changed = {op:torch._C._dispatch_dump_table('aten::'+op) != before[op]
                                           for op in before}
                                if not all(changed.values()):
                                    raise RuntimeError('Expected both unary dispatcher registrations')
                                got = getattr(torch.ops.aten, operator).default(x)
                                got_grad = getattr(torch.ops.aten, operator+'_backward').default(
                                    dy, backward_argument(operator, x, got))
                                runtime.synchronize(0)
                        else:
                            config = dict(schema=2, enabled=args.mode!='off', require_target=args.mode=='on',
                                operators={operator:dict(shapes=[list(shape)], dtypes=[str(dtype)])}, records=None)
                            policies, startup = prepare_policies(config,records,actual)
                            adapter = ElementwiseAdapter(policies,actual,runtime,gems)
                            with adapter:
                                got = forward(x)
                                got_grad, = torch.autograd.grad(got,x,dy)
                                runtime.synchronize(0)
                            item['startup'] = startup
                        delta = hits.counts - previous_hits
                        target = args.stage=='qualification' or args.mode=='on'
                        if target:
                            if not delta['GEMS '+operator.upper()+' FORWARD'] or not delta['GEMS '+operator.upper()+' BACKWARD']:
                                raise RuntimeError('Target forward/backward not both observed: '+str(dict(delta)))
                        elif delta:
                            raise RuntimeError('Native path unexpectedly entered provider')
                        detail = dict(repeat=repeat, forward_max_abs=compare(got,expected,tol if target else 0),
                            backward_max_abs=compare(got_grad,expected_grad,tol if target else 0), calls=dict(delta))
                        if adapter:
                            detail['routes'] = [dict(operator=o,backend=b,stage=s,reason=r,count=n)
                                for (o,b,s,r),n in adapter.operator_counts.items()]
                            reason = {'off':'experimental_disabled','unqualified':'capability_unverified'}.get(args.mode)
                            if reason and adapter.counts[('native','forward',reason)] != 1:
                                raise RuntimeError('Wrong native rejection reason')
                        item['repeats'].append(detail)
                    finally:
                        item['dispatcher_restored'] = before == {op:torch._C._dispatch_dump_table('aten::'+op) for op in before}
                        if not item['dispatcher_restored']:
                            raise RuntimeError('Dispatcher not restored; terminate this worker')
                    # Independent native call after scope exit must be unchanged.
                    restored = forward(x)
                    restored_grad, = torch.autograd.grad(restored,x,dy)
                    compare(restored,expected,0); compare(restored_grad,expected_grad,0)
                item['status'] = 'pass'
                print(json.dumps(item),flush=True)
    finally:
        report['observed_provider_calls'] = dict(hits.counts)
        logger.removeHandler(hits)
    report['status'] = 'pass'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--stage', choices=['inspect','qualification','routes'], required=True)
    p.add_argument('--mode', choices=['off','on','unqualified'], default='off')
    p.add_argument('--records', type=Path)
    p.add_argument('--operator', choices=tuple(OP_SHAPES), default='silu')
    p.add_argument('--shape-profile', choices=('default','onerec16'), default='default')
    a = p.parse_args()
    if a.out.exists() or (a.mode=='on' and not a.records) or (a.mode=='unqualified' and a.records):
        p.error('New output and explicit reviewed records only for qualified runs')
    report = dict(status='error', stage=a.stage, mode=a.mode,
                  scope='single-card isolated unary; not model or full three-level fallback acceptance')
    try:
        run(a,report)
    except Exception:
        report['error'] = traceback.format_exc()
        report['status'] = 'fail' if report.get('cases') else 'error'
        print(report['error'],flush=True)
    finally:
        a.out.write_text(json.dumps(report,indent=2)+'\n')
    print('overall:',report['status'].upper(),flush=True)
    return {'pass':0,'fail':1,'error':2}[report['status']]


if __name__ == '__main__':
    raise SystemExit(main())
