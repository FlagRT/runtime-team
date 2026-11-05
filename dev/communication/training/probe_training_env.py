"""Consumer preflight: verify loaded override and two usable runtime devices."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'device-context/prototype'))
import runtime
import torch
import flagcx

runtime.use('ascend')
package = Path(flagcx.__file__).resolve()
if not package.is_relative_to(Path('/workspace/FlagCX')):
    raise RuntimeError(f'Expected mounted override, got {package}')
if runtime.device_count() != 2:
    raise RuntimeError(f'Expected exactly two visible devices, got {runtime.device_count()}')
records = []
for ordinal in range(2):
    runtime.set_device(ordinal)
    device = torch.device(runtime.current().device_type, ordinal)
    if device.type != 'npu':
        raise RuntimeError('No CPU fallback')
    x = torch.ones((8, 8), device=device)
    y = x @ x
    runtime.synchronize(ordinal, timeout_ms=30000)
    if not bool((y == 8).all().item()):
        raise AssertionError('Probe matmul mismatch')
    records.append({'ordinal': ordinal, 'memory': runtime.memory_stats(ordinal)})
libraries = list(package.parent.glob('_C*.so')) + list(Path('/workspace/FlagCX/build/lib').glob('libflagcx.so*'))
print(json.dumps({'torch': torch.__version__, 'flagcx_path': str(package),
                  'libraries': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
                  'devices': records}, indent=2))
print('overall: PASS')
