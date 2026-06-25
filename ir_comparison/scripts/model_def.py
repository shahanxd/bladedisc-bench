"""
model_def.py — shared BERT-like model for all scripts.

SimpleBert: a lightweight transformer block replicating the structure
(not the weights) of BERT-base. Two variants are used:
  - layers=2, hidden=64, heads=8  → fast, used for IR dumps and profiling
  - layers=12, hidden=768, heads=12 → full BERT-base scale (used in early static dumps only)

The forward pass uses x.size() instead of unpacking x.shape so that
torch.export can keep batch/seq dimensions symbolic (dynamic shapes).
"""
import torch
import torch.nn as nn


class BertSelfAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads    = heads
        self.head_dim = hidden // heads
        self.scale    = self.head_dim ** -0.5
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out  = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        # x.size() instead of unpacking x.shape — keeps dims symbolic for torch.export
        S = x.size(1)
        q = self.q(x).view(-1, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(-1, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(-1, S, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(-1, S, x.size(2))
        return self.norm(self.out(out) + x)


class BertFFN(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear1 = nn.Linear(hidden, hidden * 4)
        self.linear2 = nn.Linear(hidden * 4, hidden)
        self.norm    = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.linear2(torch.nn.functional.gelu(self.linear1(x)))
        return self.norm(h + x)


class SimpleBert(nn.Module):
    """Lightweight BERT-like transformer. Identical op structure to BERT-base."""
    def __init__(self, layers=2, hidden=64, heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(BertSelfAttention(hidden, heads), BertFFN(hidden))
            for _ in range(layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def get_model_and_input(layers=12, hidden=768, heads=12, seq=32, device="cpu"):
    """Returns (model, dummy_input) for the standard BERT-base scale."""
    torch.manual_seed(42)
    model = SimpleBert(layers=layers, hidden=hidden, heads=heads).eval().to(device)
    dummy = torch.randn(1, seq, hidden, device=device)
    return model, dummy
