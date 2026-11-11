"""Metadata-only eager trace: phase-separated Python entries and ATen overloads."""
from collections import Counter
from contextlib import contextmanager
import json
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode


class Inventory:
    def __init__(self, limit=20000):
        self.rows = {}
        self.phase = 'outside'
        self.busy = False
        self.stack = []
        self.limit = limit
        self.handles = []

    def metadata(self, value):
        if isinstance(value, torch.Tensor):
            return dict(tensor=True, shape=list(value.shape), dtype=str(value.dtype),
                        device=str(value.device), layout=str(value.layout),
                        stride=list(value.stride()) if value.layout == torch.strided else None)
        if isinstance(value, (list, tuple)):
            return [self.metadata(x) for x in value]
        if isinstance(value, dict):
            return {str(k): self.metadata(v) for k, v in value.items()}
        if value is None or type(value) in (str, int, float, bool):
            return value
        return str(value) if isinstance(value, (torch.dtype, torch.device)) else type(value).__name__

    def call(self, layer, func, args, kwargs):
        if self.busy:
            return func(*args, **kwargs)
        self.busy = True
        try:
            inputs = self.metadata((args, kwargs))
        finally:
            self.busy = False
        result = func(*args, **kwargs)
        self.busy = True
        try:
            name = str(func) if layer == 'aten' else (
                getattr(func, '__module__', None) or 'torch.Tensor') + '.' + getattr(func, '__name__', str(func))
            key = json.dumps([self.phase, layer, name, inputs], sort_keys=True)
            if key not in self.rows:
                if len(self.rows) >= self.limit:
                    raise RuntimeError('Inventory limit reached; refusing incomplete trace')
                self.rows[key] = dict(phase=self.phase, layer=layer, operator=name,
                                     inputs=inputs, output=self.metadata(result), count=0,
                                     modules=[], module_samples_truncated=False)
            row = self.rows[key]
            row['count'] += 1
            module = self.stack[-1] if self.stack else '(outside module forward)'
            if module not in row['modules']:
                if len(row['modules']) < 8:
                    row['modules'].append(module)
                else:
                    row['module_samples_truncated'] = True
        finally:
            self.busy = False
        return result

    def attach(self, model):
        for name, module in model.named_modules():
            label = name + ':' + type(module).__name__
            def before(m, args, label=label): self.stack.append(label)
            def after(m, args, output, label=label):
                if not self.stack or self.stack.pop() != label:
                    raise RuntimeError('Module trace stack mismatch')
            self.handles.append(module.register_forward_pre_hook(before))
            self.handles.append(module.register_forward_hook(after, always_call=True))

    def detach(self):
        for handle in self.handles: handle.remove()
        self.handles.clear()

    @contextmanager
    def scope(self, phase):
        previous, self.phase = self.phase, phase
        try:
            with PythonTrace(self), AtenTrace(self): yield
        finally:
            self.phase = previous

    def report(self):
        rows = list(self.rows.values())
        counts = Counter()
        for row in rows: counts[(row['phase'], row['layer'])] += row['count']
        return dict(signatures=rows, call_totals=[dict(phase=p, layer=l, calls=n)
                    for (p, l), n in sorted(counts.items())],
                    signature_limit=self.limit, signatures_truncated=False,
                    tensor_values_recorded=False)


class PythonTrace(TorchFunctionMode):
    def __init__(self, inventory): self.inventory = inventory
    def __torch_function__(self, func, types, args=(), kwargs=None):
        return self.inventory.call('python', func, args, kwargs or {})


class AtenTrace(TorchDispatchMode):
    def __init__(self, inventory): super().__init__(); self.inventory = inventory
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self.inventory.call('aten', func, args, kwargs or {})
