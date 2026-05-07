"""CUDAlings 29.07 -- Part 7 Milestone: train a tiny GPT for a few steps.

Tie together Part 7 (Ch 21-25): nanoGPT + tokenizer + training. We use a
miniature 4-layer GPT on a synthetic "repeat the pattern" corpus that is
small enough to fit in seconds even on Pascal. 100 steps of AdamW must
reduce the loss by at least 50%.

This is the smallest possible "I trained a transformer" exercise.
"""

# I AM NOT DONE

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, E, H, T):
        super().__init__()
        self.H = H; self.E = E
        self.qkv = nn.Linear(E, 3 * E, bias=False)
        self.proj = nn.Linear(E, E, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
    def forward(self, x):
        B, T, E = x.shape
        qkv = self.qkv(x); q, k, v = qkv.split(E, dim=2)
        Dh = E // self.H
        q = q.view(B, T, self.H, Dh).transpose(1, 2)
        k = k.view(B, T, self.H, Dh).transpose(1, 2)
        v = v.view(B, T, self.H, Dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(Dh))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, E, H, T):
        super().__init__()
        self.ln1 = nn.LayerNorm(E)
        self.attn = CausalSelfAttention(E, H, T)
        self.ln2 = nn.LayerNorm(E)
        self.mlp = nn.Sequential(nn.Linear(E, 4*E), nn.GELU(), nn.Linear(4*E, E))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, V=8, T=16, E=32, H=4, n_layer=2):
        super().__init__()
        self.T = T
        self.tok = nn.Embedding(V, E)
        self.pos = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([Block(E, H, T) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, V, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks: x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def main():
    torch.manual_seed(0)
    V, T, B = 8, 16, 16
    # Synthetic corpus: token i is followed by (i+1) % V (predictable cycle).
    data = torch.arange(8192) % V

    def get_batch():
        idx = torch.randint(0, len(data) - T - 1, (B,))
        x = torch.stack([data[i:i+T] for i in idx])
        y = torch.stack([data[i+1:i+T+1] for i in idx])
        return x, y

    model = TinyGPT(V=V, T=T)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for step in range(100):
        x, y = get_batch()
        # TODO: forward, backward, optimizer step; record the scalar loss into `losses`
        pass

    if losses and losses[-1] < losses[0] * 0.5:
        print("ok")
    else:
        s = (losses[0], losses[-1]) if losses else (0, 0)
        print(f"FAIL start={s[0]:.3f} end={s[1]:.3f}")


if __name__ == "__main__":
    main()
