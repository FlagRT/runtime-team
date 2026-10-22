"""Host-only ordering and accounting tests, not FlagCX/NPU acceptance."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import distributed as comm


class ManualReducerTests(unittest.TestCase):
    def group(self):
        with patch.object(comm.dist, 'is_initialized', return_value=False), \
             patch.object(comm.dist, 'init_process_group'), \
             patch.object(comm.dist, 'get_backend', return_value='flagcx'):
            return comm.TrainingGroup('flagcx', 0, 2, Mock())

    def model(self):
        model = torch.nn.Linear(3, 2)
        for p in model.parameters(): p.grad = torch.ones_like(p)
        return model

    def check_mean(self, fused):
        group, model, pending = self.group(), self.model(), []
        # Simulate delayed writes; dividing before synchronize would be wrong.
        def reduce(t, **kw): pending.append(t)
        def sync():
            for t in pending: t.add_(3.)
            pending.clear()
        group.synchronize.side_effect = sync
        with patch.object(comm.dist, 'all_reduce', side_effect=reduce):
            comm.manual_mean_gradients(model, group, fused)
        for p in model.parameters(): torch.testing.assert_close(p.grad, torch.full_like(p, 2.))
        self.assertEqual(group.calls, 1 if fused else 2)
        self.assertEqual(group.payload_bytes, 32)

    def test_unfused_waits_before_mean(self): self.check_mean(False)
    def test_fused_waits_before_unpack(self): self.check_mean(True)

    def test_missing_gradient_rejected(self):
        m = self.model(); m.bias.grad = None
        with self.assertRaises(ValueError): comm.manual_mean_gradients(m, self.group())

    def test_mixed_dtype_rejected(self):
        m = torch.nn.ModuleList([torch.nn.Linear(1,1), torch.nn.Linear(1,1).double()])
        for p in m.parameters(): p.grad = torch.ones_like(p)
        with self.assertRaises(ValueError): comm.manual_mean_gradients(m, self.group(), True)

    def test_failure_blocks_reuse(self):
        g = self.group(); failure = TimeoutError('injected')
        g.synchronize.side_effect = failure
        with patch.object(comm.dist, 'all_reduce') as reduce:
            with self.assertRaises(TimeoutError): comm.manual_mean_gradients(self.model(), g)
            self.assertEqual(g.state, 'FAILED')
            with self.assertRaises(RuntimeError): comm.manual_mean_gradients(self.model(), g)
            reduce.assert_not_called()

    def test_post_collective_failure_blocks_reuse(self):
        g = self.group(); g.synchronize.side_effect = [None, TimeoutError('post wait')]
        with patch.object(comm.dist, 'all_reduce') as reduce:
            with self.assertRaises(TimeoutError): comm.manual_mean_gradients(self.model(), g)
            self.assertEqual(g.state, 'FAILED')
            with self.assertRaises(RuntimeError): comm.manual_mean_gradients(self.model(), g)
            reduce.assert_called_once()

    def test_initial_state_broadcast_is_synchronized(self):
        g = self.group(); m = self.model(); order = []
        g.synchronize.side_effect = lambda: order.append('sync')
        with patch.object(comm.dist, 'broadcast', side_effect=lambda *a,**k: order.append('broadcast')):
            comm.ManualDataParallel(m, g)
        self.assertEqual(order, ['sync', 'broadcast', 'sync', 'sync', 'broadcast', 'sync'])

    def test_buckets_split_large_parameter_and_tail(self):
        for cap in (4, 12, 16, 28, 32, 100):
            with self.subTest(cap=cap):
                group, model, sizes = self.group(), self.model(), []
                def reduce(t, **kw):
                    sizes.append(t.numel()*t.element_size())
                    t.add_(3.)
                with patch.object(comm.dist, 'all_reduce', side_effect=reduce):
                    comm.manual_mean_gradients(model, group, bucket_bytes=cap)
                for p in model.parameters():
                    torch.testing.assert_close(p.grad, torch.full_like(p, 2.))
                self.assertLessEqual(max(sizes), cap)
                self.assertEqual(sum(sizes), 32)
                self.assertEqual(group.calls, (32+cap-1)//cap)

    def test_buckets_wait_for_delayed_collective(self):
        group, model, pending = self.group(), self.model(), []
        def sync():
            for tensor in pending: tensor.add_(3.)
            pending.clear()
        group.synchronize.side_effect = sync
        with patch.object(comm.dist, 'all_reduce', side_effect=lambda t, **kw: pending.append(t)):
            comm.manual_mean_gradients(model, group, bucket_bytes=12)
        for p in model.parameters():
            torch.testing.assert_close(p.grad, torch.full_like(p, 2.))

    def test_invalid_bucket_rejected_before_collective(self):
        for cap in (0, -1, 3, True, 4.0):
            with self.subTest(cap=cap), patch.object(comm.dist, 'all_reduce') as reduce:
                with self.assertRaises(ValueError):
                    comm.manual_mean_gradients(self.model(), self.group(), bucket_bytes=cap)
                reduce.assert_not_called()

    def test_bucket_and_fused_conflict(self):
        with self.assertRaises(ValueError):
            comm.manual_mean_gradients(self.model(), self.group(), True, bucket_bytes=16)

    def test_bucket_noncontiguous_rejected(self):
        model = torch.nn.Linear(2, 2, bias=False)
        model.weight.grad = torch.ones(2, 2).t()
        with self.assertRaises(ValueError):
            comm.manual_mean_gradients(model, self.group(), bucket_bytes=16)

    def test_bucket_failure_is_terminal(self):
        group = self.group()
        with patch.object(comm.dist, 'all_reduce', side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                comm.manual_mean_gradients(self.model(), group, bucket_bytes=12)
        self.assertEqual(group.state, 'FAILED')

    def test_bucket_real_values_and_unaligned_cap(self):
        group, model = self.group(), self.model()
        original = torch.arange(8, dtype=torch.float32) / 3
        peer = torch.arange(8, dtype=torch.float32).flip(0) / 7
        offset = 0
        for p in model.parameters():
            p.grad = original[offset:offset+p.numel()].reshape_as(p).clone()
            offset += p.numel()
        offset = 0
        def reduce(t, **kw):
            nonlocal offset
            t.add_(peer[offset:offset+t.numel()])
            offset += t.numel()
        with patch.object(comm.dist, 'all_reduce', side_effect=reduce), \
             patch.object(comm.torch, 'empty', wraps=torch.empty) as allocate:
            comm.manual_mean_gradients(model, group, bucket_bytes=9)
        allocate.assert_called_once()
        self.assertEqual(allocate.call_args.args[0], 2)
        actual = torch.cat([p.grad.flatten() for p in model.parameters()])
        torch.testing.assert_close(actual, (original+peer)/2)
        self.assertEqual(group.calls, 4)
        self.assertEqual(group.payload_bytes, 32)


if __name__ == '__main__': unittest.main()
