"""CPU wiring only: fake registration does not test any FlagGems kernel."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
import torch
import torch.nn.functional as F
import _adapter_package
from flagrt_adapter.bridge import ElementwiseAdapter
from flagrt_adapter.policy import Environment, OperatorPolicy


class MultiOperatorTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        @contextmanager
        def registration(include):
            self.calls.append(tuple(include))
            yield
        self.gems = SimpleNamespace(use_gems=registration)
        self.runtime = SimpleNamespace(synchronize=lambda _: None)
        self.env = Environment(*(["synthetic"] * 7))

    def adapter(self, names):
        # Explicit CPU-only injection; never produce a capability record from it.
        policies = {name: SimpleNamespace(operator=name, reason=lambda *_: None) for name in names}
        return ElementwiseAdapter(policies, self.env, self.runtime, self.gems)

    def test_mixed_forward_backward(self):
        x = torch.linspace(-8, 8, 24).reshape(4, 6).requires_grad_()
        dy = torch.randn_like(x)
        expected = torch.sigmoid(F.silu(x))
        expected_grad, = torch.autograd.grad(expected, x, dy)
        ops = ("silu", "silu_backward", "sigmoid", "sigmoid_backward")
        before = {op: torch._C._dispatch_dump_table("aten::" + op) for op in ops}
        with self.adapter(("silu", "sigmoid")) as adapter:
            result = torch.sigmoid(F.silu(x))
            grad, = torch.autograd.grad(result, x, dy)
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)
        self.assertEqual(len(self.calls), 4)
        for name in ("silu", "sigmoid"):
            for stage in ("forward", "backward"):
                self.assertEqual(adapter.operator_counts[(name, "flaggems", stage, "completed")], 1)
        self.assertEqual(before, {op: torch._C._dispatch_dump_table("aten::" + op) for op in ops})

    def test_sigmoid_public_variants(self):
        for func in (torch.sigmoid, F.sigmoid):
            with self.subTest(func=func):
                x = torch.randn(2, 3, requires_grad=True)
                dy = torch.randn_like(x)
                expected = func(x)
                expected_grad, = torch.autograd.grad(expected, x, dy)
                with self.adapter(("sigmoid",)):
                    result = func(input=x)
                    grad, = torch.autograd.grad(result, x, dy)
                torch.testing.assert_close(result, expected, atol=0, rtol=0)
                torch.testing.assert_close(grad, expected_grad, atol=0, rtol=0)

    def test_absent_operator_stays_native(self):
        with self.adapter(("silu",)) as adapter:
            torch.sigmoid(torch.randn(2, 3, requires_grad=True)).sum().backward()
        self.assertFalse(self.calls)
        self.assertEqual(adapter.operator_counts[("sigmoid", "native", "forward", "operator_disabled")], 1)

    def test_sigmoid_out_stays_native(self):
        x = torch.randn(2, 3)
        out = torch.empty_like(x)
        with self.adapter(("sigmoid",)) as adapter:
            result = torch.sigmoid(x, out=out)
        self.assertIs(result, out)
        self.assertFalse(self.calls)
        self.assertEqual(adapter.operator_counts[("sigmoid", "native", "forward", "signature")], 1)

    def test_policy_cannot_be_used_for_another_operator(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ElementwiseAdapter({"sigmoid": OperatorPolicy(self.env, ((2, 3),))},
                               self.env, self.runtime, self.gems)

    def test_failed_sigmoid_aborts_scope(self):
        def fail(_):
            raise RuntimeError("injected failure")
        self.runtime.synchronize = fail
        adapter = self.adapter(("silu", "sigmoid"))
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with adapter:
                torch.sigmoid(torch.randn(2, 3))
        self.assertTrue(adapter.failed)
        self.assertFalse(self.calls)
        self.assertEqual(adapter.operator_counts[("sigmoid", "flaggems", "forward", "error_no_retry")], 1)


if __name__ == "__main__":
    unittest.main()
