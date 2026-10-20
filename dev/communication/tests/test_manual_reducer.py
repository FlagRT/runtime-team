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


if __name__ == '__main__': unittest.main()
