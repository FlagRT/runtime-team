"""Self-contained P800 GEMM/Epilogue reproduction with an isolated compiler."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['original', 'gemm', 'epilogue'], required=True)
    parser.add_argument('--epi', type=int, choices=range(4), default=3,
                        help='0=GEMM, 1=Bias, 2=ReLU, 3=Bias+ReLU')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['TRITON_CACHE_DIR'] = str(args.output.resolve() / 'cache')
    os.environ['XPU_FIX_IR_DIR'] = str(args.output.resolve() / 'ir')
    import numpy as np
    import torch
    import triton
    from kernel import matmul
    from prepare import prepare
    backend = importlib.import_module('triton.backends.xpu.compiler')
    original = Path(backend.__file__).read_bytes()
    record = dict(mode=args.mode, epi=args.epi, torch=torch.__version__, triton=triton.__version__,
                  target=str(triton.runtime.driver.active.get_current_target()),
                  device=str(torch.cuda.get_device_properties(0)),
                  installed_compiler_sha256=hashlib.sha256(original).hexdigest(), samples=[])
    try:
        if triton.runtime.driver.active.get_current_target().backend != 'xpu':
            raise RuntimeError('This reproduction requires a real XPU backend')
        if args.mode != 'original':
            data = prepare(original, args.mode)
            record['modified_compiler_sha256'] = hashlib.sha256(data).hexdigest()
            copy = args.output / 'compiler_copy.py'
            copy.write_bytes(data)
            spec = importlib.util.spec_from_file_location('isolated_sdnn_compiler', copy)
            patch = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = patch
            spec.loader.exec_module(patch)
            patch.__file__ = backend.__file__  # Keep installed SDK tool/bitcode paths.
            live = backend.XPUBackend
            methods = {name: getattr(patch.XPUBackend, name) for name in ('make_ttxir', 'make_llir')}
            patch.XPUBackend = live
            for name, method in methods.items():
                setattr(live, name, staticmethod(method))
        rng = np.random.default_rng(20261019)
        for m, k, n in [(128, 128, 128), (77, 131, 193), (160, 96, 80)]:
            for kind in ('random', 'zero', 'constant', 'mixed_scale'):
                a = rng.uniform(-0.25, 0.25, (m, k)).astype(np.float32)
                b = rng.uniform(-0.25, 0.25, (k, n)).astype(np.float32)
                bias = rng.uniform(-0.25, 0.25, n).astype(np.float32)
                if kind == 'zero':
                    a.fill(0); b.fill(0); bias.fill(0)
                elif kind == 'constant':
                    a.fill(0.25); b.fill(0.25); bias.fill(0.25)
                elif kind == 'mixed_scale':
                    a[::2] *= 0.001
                    b[::2] *= 2
                reference = a.astype(np.float64) @ b.astype(np.float64)
                if args.epi in (1, 3):
                    reference += bias.astype(np.float64)[None, :]
                if args.epi in (2, 3):
                    reference = np.maximum(reference, 0)
                ta, tb, tc = [torch.from_numpy(x).to('cuda') for x in (a, b, bias)]
                output = torch.full((m, n), float('nan'), device='cuda', dtype=torch.float32)
                compiled = matmul[(triton.cdiv(m, 64)*triton.cdiv(n, 64),)](
                    ta, tb, tc, output, m, k, n, BM=64, BN=64, BK=32, EPI=args.epi)
                torch.cuda.synchronize()
                actual = output.cpu().numpy().astype(np.float64)
                error = np.abs(actual-reference)
                relative = float(np.linalg.norm(error)/max(np.linalg.norm(reference), 1e-12))
                passed = bool(np.isfinite(actual).all() and (error <= 1e-4+1e-4*np.abs(reference)).all()
                              and relative <= 1e-4)
                name = f'{m}x{k}x{n}-{kind}'
                np.savez(args.output / (name + '.npz'), a=a, b=b, bias=bias,
                         actual=actual, reference=reference)
                record['samples'].append(dict(shape=[m, k, n], kind=kind, passed=passed,
                                               max_abs_error=float(error.max()), relative_l2=relative))
                for ext, content in compiled.asm.items():
                    if isinstance(content, (str, bytes)):
                        (args.output / f'{m}x{k}x{n}.{ext}').write_bytes(
                            content.encode() if isinstance(content, str) else content)
                if not passed:
                    raise AssertionError('Device output differs from FP64 reference')
        record['status'] = 'passed'
    except Exception:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        print(record['error'])
    if Path(backend.__file__).read_bytes() != original:
        raise AssertionError('Installed compiler changed')
    (args.output / 'result.json').write_text(json.dumps(record, indent=2))
    print('FINAL', record['status'], 'SAMPLES', len(record['samples']))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
