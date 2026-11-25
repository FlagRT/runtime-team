import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from priority_models import MODEL_NAMES, make_batch, make_case


class PriorityModelTests(unittest.TestCase):
    def test_din_empty_history_backpropagates_finite_gradients(self):
        model, loss = make_case('din')
        output = model(torch.zeros(2, 6, dtype=torch.long), torch.tensor([1, 2]))
        self.assertEqual(output.shape, (2,))
        loss(output, torch.tensor([0., 1.])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(torch.equal(model.items.weight.grad[0], torch.zeros(16)))

    def test_din_padding_does_not_change_score(self):
        torch.manual_seed(9)
        model, _ = make_case('din')
        q = torch.tensor([4, 5])
        x = torch.tensor([[1, 2], [3, 4]])
        torch.testing.assert_close(model(x, q), model(torch.nn.functional.pad(x, (0, 4)), q))

    def test_dssm_positive_labels_and_both_towers(self):
        model, loss = make_case('dssm')
        (users, items), indices = make_batch('dssm', 16, 32100)
        self.assertTrue(torch.equal(users, items[torch.arange(16), indices, :3]))
        scores = model(users, items)
        self.assertEqual(scores.shape, (16, 4))
        loss(scores, indices).backward()
        for tower in (model.user, model.item):
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in tower.parameters()))

    def test_seeded_samples_reproducible(self):
        for name in MODEL_NAMES:
            x, y = make_batch(name, 16, 987)
            xx, yy = make_batch(name, 16, 987)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(x, xx)))
            self.assertTrue(torch.equal(y, yy))

    def test_dcn_mix_backward_reaches_gates_and_every_expert(self):
        model, loss = make_case('dcn-mix')
        inputs, y = make_batch('dcn-mix', 16, 7)
        scores = model(*inputs)
        self.assertEqual(scores.shape, (16,))
        loss(scores, y).backward()
        for block in (*model.cross, *model.gates, model.deep):
            for p in block.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0.)

    def test_sasrec_causal_prefix_and_next_item_targets(self):
        model, loss = make_case('sasrec')
        (history,), y = make_batch('sasrec', 16, 7)
        self.assertTrue(torch.equal(history[:, 1:], y[:, :-1]))
        updated = history.clone()
        updated[:, 3:] = (updated[:, 3:] + 11) % 64
        scores = model(history)
        self.assertEqual(scores.shape, (16, 6, 64))
        torch.testing.assert_close(scores[:, :3], model(updated)[:, :3])
        loss(scores, y).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
