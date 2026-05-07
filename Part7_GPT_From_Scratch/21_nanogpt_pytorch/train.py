"""Train the nanoGPT in this chapter on Tiny Shakespeare (char-level).

Why char-level? Vocab is 65 characters. No tokenizer dependency, the
loss curve is interpretable, and on a Pascal GPU we can train a 1M
parameter model in 5 minutes. Plenty to see the model learn punctuation,
word boundaries, and rough Shakespeare cadence.

Run:
    python train.py            # ~5 min, saves ckpt.pt
"""

from __future__ import annotations

import os
import time
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).parent / "input.txt"


def load_text() -> str:
    if not DATA_PATH.exists():
        print(f"downloading {DATA_URL} → {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def build_dataset(text: str):
    """Char-level tokenizer + train/val split.
    Returns (data_train, data_val, stoi, itos)."""
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(0.9 * len(data))
    return data[:n], data[n:], stoi, itos


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    """Sample (B, T) input + target chunks. Targets are shifted by one."""
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i:i + block_size]     for i in ix])
    y = np.stack([data[i+1:i + block_size+1] for i in ix])
    x = torch.from_numpy(x).to(device, non_blocking=True)
    y = torch.from_numpy(y).to(device, non_blocking=True)
    return x, y


# ---------------------------------------------------------------------------
# Hyperparams
# ---------------------------------------------------------------------------
HP = dict(
    block_size = 128,
    batch_size = 32,
    n_layer    = 4,
    n_head     = 4,
    n_embd     = 128,
    dropout    = 0.0,

    lr         = 3e-4,
    max_iters  = 3000,
    eval_iters = 100,
    eval_every = 500,
    grad_clip  = 1.0,
    weight_decay = 0.1,
    betas      = (0.9, 0.95),

    seed       = 1337,
    device     = "cuda" if torch.cuda.is_available() else "cpu",
)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model: GPT, train_data, val_data, hp) -> dict[str, float]:
    """Run eval_iters mini-batches on each split and return mean losses."""
    model.train(False)
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(hp["eval_iters"])
        for k in range(hp["eval_iters"]):
            x, y = get_batch(data, hp["block_size"], hp["batch_size"], hp["device"])
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train(True)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    hp = HP
    torch.manual_seed(hp["seed"])
    np.random.seed(hp["seed"])

    text = load_text()
    train_data, val_data, stoi, itos = build_dataset(text)
    print(f"loaded {len(text):,} chars, vocab={len(stoi)}, "
          f"train={len(train_data):,} val={len(val_data):,}")

    cfg = GPTConfig(
        block_size = hp["block_size"],
        vocab_size = len(stoi),
        n_layer    = hp["n_layer"],
        n_head     = hp["n_head"],
        n_embd     = hp["n_embd"],
        dropout    = hp["dropout"],
    )
    model = GPT(cfg).to(hp["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M parameters")

    # Karpathy's nanoGPT trick: weight decay on 2D params (matmul weights)
    # but NOT on biases or layernorm/embedding params.
    decay = [p for n, p in model.named_parameters()
             if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters()
                if p.requires_grad and p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": hp["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=hp["lr"], betas=hp["betas"],
    )

    best_val = float("inf")
    t0 = time.time()
    for step in range(hp["max_iters"] + 1):
        if step % hp["eval_every"] == 0 or step == hp["max_iters"]:
            losses = estimate_loss(model, train_data, val_data, hp)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} "
                  f"val {losses['val']:.4f} | {elapsed:.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(),
                            "cfg": cfg, "stoi": stoi, "itos": itos},
                           Path(__file__).parent / "ckpt.pt")

        x, y = get_batch(train_data, hp["block_size"], hp["batch_size"], hp["device"])
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        optim.step()

    print(f"done. best val loss = {best_val:.4f}. ckpt.pt saved.")


if __name__ == "__main__":
    main()
