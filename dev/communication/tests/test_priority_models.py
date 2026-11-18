import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from priority_models import make_batch, make_case


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
        for name in ('din', 'dssm'):
            x, y = make_batch(name, 16, 987)
            xx, yy = make_batch(name, 16, 987)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(x, xx)))
            self.assertTrue(torch.equal(y, yy))
