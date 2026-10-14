"""Torch CPU controls. Fake registration tests wiring, NOT FlagGems/NPU kernels."""
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
import unittest
import threading
import torch
import torch.nn.functional as F
from silu_policy import Environment, SiluPolicy
from silu_training_adapter import SiluAdapter


class Controls(unittest.TestCase):
    def setUp(self):
        self.env = Environment("test", "test", "test", "test")
        self.policy = SiluPolicy(self.env, ((2, 3),))
        self.calls = []

        @contextmanager
        def registration(**kwargs):
            self.calls.append("enter")
            try:
                yield
            finally:
                self.calls.append("exit")
        self.gems = SimpleNamespace(use_gems=registration)
        self.runtime = SimpleNamespace(synchronize=lambda index: None)

    def adapter(self, forced=False):
        policy = self.policy
        if forced:
            # Test seam ONLY: bypass NPU admission to exercise autograd on CPU.
            policy = SimpleNamespace(reason=lambda *_: None)
        return SiluAdapter(policy, self.env, self.runtime, self.gems)

    def test_default_native(self):
        x = torch.randn(2, 3, requires_grad=True)
        with self.adapter() as scope:
            F.silu(x).sum().backward()
        self.assertFalse(self.calls)
        self.assertEqual(scope.counts[("native", "forward", "experimental_disabled")], 1)

    def test_forward_backward_restore(self):
        x = torch.randn(2, 3, requires_grad=True)
        dy = torch.randn_like(x)
        expected = F.silu(x)
        expected_grad, = torch.autograd.grad(expected, x, dy)
        with self.adapter(True) as scope:
            y = F.silu(x)
            grad, = torch.autograd.grad(y, x, dy)
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(grad, expected_grad)
        self.assertEqual(self.calls, ["enter", "exit", "enter", "exit"])
        self.assertFalse(scope.failed)

    def test_body_error_exit(self):
        scope = self.adapter()
        with self.assertRaisesRegex(ValueError, "body"):
            with scope:
                raise ValueError("body")
        self.assertFalse(scope.entered)
        with self.assertRaises(RuntimeError):
            with scope:
                pass
        with self.adapter():
            pass

    def test_nested_rejected(self):
        with self.adapter():
            with self.assertRaises(RuntimeError):
                with self.adapter():
                    pass

    def test_backward_outside_scope(self):
        x = torch.randn(2, 3, requires_grad=True)
        with self.adapter(True):
            y = F.silu(x)
        with self.assertRaises(RuntimeError):
            y.backward(torch.ones_like(y))

    def test_target_failure_not_retried(self):
        def fail(_):
            raise RuntimeError("device failure")
        self.runtime.synchronize = fail
        scope = self.adapter(True)
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            with scope:
                F.silu(torch.randn(2, 3))
        self.assertTrue(scope.failed)
        self.assertFalse(self.calls)
        self.assertEqual(scope.counts[("flaggems", "forward", "error_no_retry")], 1)

    def test_registration_exit_failure(self):
        @contextmanager
        def broken(**kwargs):
            self.calls.append("enter")
            try:
                yield
            finally:
                self.calls.append("exit")
                raise RuntimeError("registration exit failed")
        self.gems.use_gems = broken
        scope = self.adapter(True)
        with self.assertRaisesRegex(RuntimeError, "registration exit failed"):
            with scope:
                F.silu(torch.randn(2, 3))
        self.assertTrue(scope.failed)
        self.assertEqual(self.calls, ["enter", "exit"])

    def test_higher_order_rejected(self):
        x = torch.randn(2, 3, requires_grad=True)
        scope = self.adapter(True)
        with self.assertRaisesRegex(RuntimeError, "Higher-order"):
            with scope:
                torch.autograd.grad(F.silu(x), x, torch.ones_like(x), create_graph=True)
        self.assertTrue(scope.failed)

    def test_inference(self):
        x = torch.randn(2, 3)
        expected = F.silu(x)
        with self.adapter(True), torch.inference_mode():
            actual = F.silu(x)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(self.calls, ["enter", "exit"])

    def test_engine_thread_allowed_but_foreign_forward_rejected(self):
        results, errors = [], []
        x = torch.randn(2, 3)
        scope = self.adapter(True)
        def worker():
            try:
                scope.check()  # Foreign forward is still rejected.
            except RuntimeError:
                results.append("forward_rejected")
            try:
                results.append(scope.target("backward", x, torch.ones_like(x)))
            except BaseException as exc:
                errors.append(exc)
        with scope:
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
        self.assertFalse(errors)
        self.assertEqual(results[0], "forward_rejected")
        torch.testing.assert_close(results[1], torch.ops.aten.silu_backward.default(torch.ones_like(x), x))

    def test_backward_failure_no_update(self):
        x = torch.randn(2, 3, requires_grad=True)
        before = x.detach().clone()
        optimizer = torch.optim.SGD([x], lr=0.1)
        scope = self.adapter(True)
        def fail(_):
            raise RuntimeError("backward device failure")
        with self.assertRaisesRegex(RuntimeError, "backward device failure"):
            with scope:
                y = F.silu(x)
                self.runtime.synchronize = fail
                y.backward(torch.ones_like(y))
                optimizer.step()
        torch.testing.assert_close(x, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
