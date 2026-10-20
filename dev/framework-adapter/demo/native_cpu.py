"""No accelerator or FlagGems needed: demonstrate fail-closed native operation."""
import json
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from flagrt_adapter import Environment, OperatorPolicy
from flagrt_adapter.bridge import ElementwiseAdapter


def main():
    def forbidden(*args, **kwargs):
        raise AssertionError("Default native demo must never register or synchronize target")
    env = Environment("cpu-demo", torch.__version__, "", "")
    policies = {name: OperatorPolicy(env, ((2, 3),), operator=name)
                for name in ("silu", "sigmoid")}
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    dy = torch.ones_like(x)
    reference = torch.sigmoid(F.silu(x))
    reference_grad, = torch.autograd.grad(reference, x, dy)
    with ElementwiseAdapter(policies, env, SimpleNamespace(synchronize=forbidden),
                            SimpleNamespace(use_gems=forbidden)) as adapter:
        y = torch.sigmoid(F.silu(x))
        grad, = torch.autograd.grad(y, x, dy)
    torch.testing.assert_close(y, reference, atol=0, rtol=0)
    torch.testing.assert_close(grad, reference_grad, atol=0, rtol=0)
    assert len(adapter.events) == 2
    assert all(e["backend"] == "native" and e["reason"] == "experimental_disabled"
               for e in adapter.events)
    print(json.dumps({"overall": "PASS", "scope": "CPU native control only",
                      "events": list(adapter.events)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
