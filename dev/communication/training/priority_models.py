"""Minimal synthetic training fixtures, not pretrained models or quality benchmarks."""
import torch
from torch import nn
from torch.nn import functional as F


class DIN(nn.Module):
    """Candidate-aware history attention, with padding excluded from aggregation."""
    def __init__(self):
        super().__init__()
        self.items = nn.Embedding(128, 16, padding_idx=0)
        self.attention = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, history, candidate):
        h = self.items(history)
        q = self.items(candidate).unsqueeze(1).expand_as(h)
        scores = self.attention(torch.cat((h, q, h-q, h*q), dim=-1)).squeeze(-1)
        valid = history.ne(0)
        weights = torch.softmax(scores.masked_fill(~valid, -1e4), dim=1) * valid
        interest = (h * weights.unsqueeze(-1)).sum(dim=1)
        return self.head(torch.cat((interest, q[:, 0]), dim=-1)).squeeze(-1)


class DSSM(nn.Module):
    """User/item towers with cosine logits and explicit per-example negatives."""
    def __init__(self):
        super().__init__()
        def tower(width):
            return nn.Sequential(nn.Embedding(128, 16), nn.Flatten(-2),
                                 nn.Linear(width * 16, 24), nn.Tanh(), nn.Linear(24, 12))
        self.user = tower(3)
        self.item = tower(4)

    def forward(self, users, items):
        user = F.normalize(self.user(users), dim=-1)
        item = F.normalize(self.item(items), dim=-1)
        return (user.unsqueeze(1) * item).sum(dim=-1) * 8


def make_case(name):
    if name == 'din':
        return DIN(), F.binary_cross_entropy_with_logits
    if name == 'dssm':
        return DSSM(), F.cross_entropy
    raise ValueError(f'Unknown model: {name}')


def make_batch(name, size, seed):
    g = torch.Generator().manual_seed(seed)
    if name == 'din':
        candidate = torch.randint(1, 128, (size,), generator=g)
        history = torch.randint(1, 128, (size, 6), generator=g)
        length = torch.randint(0, 7, (size,), generator=g)
        history[:, 0] = torch.where(torch.arange(size) % 2 == 0, candidate, history[:, 0])
        history.masked_fill_(torch.arange(6) >= length[:, None], 0)
        target = history.eq(candidate[:, None]).any(dim=1).float()
        return (history, candidate), target
    if name == 'dssm':
        users = torch.randint(1, 128, (size, 3), generator=g)
        items = torch.randint(1, 128, (size, 4, 4), generator=g)
        target = torch.randint(0, 4, (size,), generator=g)
        items[torch.arange(size), target, :3] = users
        return (users, items), target
    raise ValueError(f'Unknown model: {name}')
