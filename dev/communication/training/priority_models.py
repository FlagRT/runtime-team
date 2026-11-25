"""Minimal synthetic training fixtures, not pretrained models or quality benchmarks."""
import torch
from torch import nn
from torch.nn import functional as F

MODEL_NAMES = ('din', 'dssm', 'dcn-mix', 'sasrec')


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


class DCNMix(nn.Module):
    """Small CTR fixture with gated low-rank cross experts and a deep branch."""
    def __init__(self):
        super().__init__()
        width, rank, experts = 20, 4, 3
        self.fields = nn.Embedding(64, 4)
        self.cross = nn.ModuleList(
            nn.ModuleList(nn.Sequential(nn.Linear(width, rank, bias=False), nn.Tanh(),
                                        nn.Linear(rank, rank, bias=False), nn.Tanh(),
                                        nn.Linear(rank, width)) for _ in range(experts))
            for _ in range(2))
        self.gates = nn.ModuleList(nn.Linear(width, experts) for _ in self.cross)
        self.deep = nn.Sequential(nn.Linear(width, 16), nn.ReLU())
        self.head = nn.Linear(width + 16, 1)

    def forward(self, fields, dense):
        x0 = torch.cat((self.fields(fields).flatten(1), dense), dim=-1)
        x = x0
        for experts, gate in zip(self.cross, self.gates, strict=True):
            weights = gate(x).softmax(dim=-1).unsqueeze(-1)
            mixed = (weights * torch.stack([expert(x) for expert in experts], dim=1)).sum(1)
            x = x + x0 * mixed
        return self.head(torch.cat((x, self.deep(x0)), dim=-1)).squeeze(-1)


class SASRec(nn.Module):
    """One causal attention block for next-item prediction; no pretrained weights."""
    def __init__(self):
        super().__init__()
        width = 16
        self.items = nn.Embedding(64, width)
        self.positions = nn.Embedding(6, width)
        self.qkv = nn.Linear(width, 3 * width)
        self.project = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 24), nn.ReLU(), nn.Linear(24, width))
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 64)
        self.register_buffer('future', torch.ones(6, 6, dtype=torch.bool).triu(1),
                             persistent=False)

    def forward(self, history):
        n = history.size(1)
        x = self.items(history) + self.positions(torch.arange(n, device=history.device))
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) * 0.25
        weights = scores.masked_fill(self.future[:n, :n], -1e4).softmax(dim=-1)
        x = self.attention_norm(x + self.project(weights @ v))
        return self.head(self.output_norm(x + self.feedforward(x)))


def next_item_loss(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def make_case(name):
    if name == 'din':
        return DIN(), F.binary_cross_entropy_with_logits
    if name == 'dssm':
        return DSSM(), F.cross_entropy
    if name == 'dcn-mix':
        return DCNMix(), F.binary_cross_entropy_with_logits
    if name == 'sasrec':
        return SASRec(), next_item_loss
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
    if name == 'dcn-mix':
        fields = torch.randint(0, 64, (size, 3), generator=g)
        dense = torch.randn(size, 8, generator=g)
        target = ((fields[:, 0] % 2 == fields[:, 1] % 2) ^ (dense[:, 0] > 0)).float()
        return (fields, dense), target
    if name == 'sasrec':
        start = torch.randint(0, 64, (size, 1), generator=g)
        stride = torch.randint(1, 5, (size, 1), generator=g)
        sequence = (start + stride * torch.arange(7)) % 64
        return (sequence[:, :-1].contiguous(),), sequence[:, 1:].contiguous()
    raise ValueError(f'Unknown model: {name}')
