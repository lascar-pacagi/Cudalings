"""Train a 10M-param BPE GPT with a real LR schedule.

Improvements vs chapter 21's train.py:
    - BPE tokens (vocab=1024) instead of chars
    - Linear warmup → cosine decay LR schedule
    - Gradient accumulation (so effective batch can exceed GPU memory)
    - Best-checkpoint tracking by val loss
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

# Import the model from chapter 21 (sibling chapter directory).
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "21_nanogpt_pytorch"))
from model import GPT, GPTConfig                                  # noqa: E402

from tokenizer_bpe import BPETokenizer
from data import open_shard, get_batch


# ---------------------------------------------------------------------------
# Hyperparams (Quadro P4200 friendly)
# ---------------------------------------------------------------------------
HP = dict(
    block_size = 256,
    micro_batch = 8,             # batch per fwd/bwd pass
    accum_steps = 4,             # effective batch = micro_batch * accum_steps = 32
    n_layer    = 6,
    n_head     = 6,
    n_embd     = 384,
    dropout    = 0.0,

    lr_max     = 3e-4,
    lr_min     = 3e-5,
    warmup     = 100,
    max_iters  = 5000,
    grad_clip  = 1.0,
    weight_decay = 0.1,
    betas      = (0.9, 0.95),

    eval_iters = 50,
    eval_every = 500,

    seed       = 1337,
    device     = "cuda" if torch.cuda.is_available() else "cpu",
)


def lr_at(step: int, hp) -> float:
    """Linear warmup then cosine decay from lr_max → lr_min."""
    if step < hp["warmup"]:
        return hp["lr_max"] * (step + 1) / hp["warmup"]
    progress = (step - hp["warmup"]) / max(1, hp["max_iters"] - hp["warmup"])
    progress = min(1.0, progress)
    return hp["lr_min"] + 0.5 * (hp["lr_max"] - hp["lr_min"]) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, hp):
    model.train(False)
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        L = torch.zeros(hp["eval_iters"])
        for k in range(hp["eval_iters"]):
            x, y = get_batch(data, hp["block_size"], hp["micro_batch"], hp["device"])
            _, loss = model(x, y)
            L[k] = loss.item()
        out[split] = L.mean().item()
    model.train(True)
    return out


def main():
    hp = HP
    torch.manual_seed(hp["seed"])

    tok = BPETokenizer.load(HERE / "bpe.json")
    train_data = open_shard(HERE / "train.bin")
    val_data   = open_shard(HERE / "val.bin")
    print(f"vocab={tok.vocab_size}  train_tokens={len(train_data):,}  val={len(val_data):,}")

    cfg = GPTConfig(
        block_size=hp["block_size"], vocab_size=tok.vocab_size,
        n_layer=hp["n_layer"], n_head=hp["n_head"], n_embd=hp["n_embd"],
        dropout=hp["dropout"],
    )
    model = GPT(cfg).to(hp["device"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M parameters")

    decay   = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay,    "weight_decay": hp["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=hp["lr_max"], betas=hp["betas"],
    )

    best_val = float("inf")
    t0 = time.time()
    for step in range(hp["max_iters"] + 1):
        # set LR for this step on every param group
        lr = lr_at(step, hp)
        for g in optim.param_groups:
            g["lr"] = lr

        if step % hp["eval_every"] == 0:
            losses = estimate_loss(model, train_data, val_data, hp)
            elapsed = time.time() - t0
            print(f"step {step:5d}  lr {lr:.5f}  "
                  f"train {losses['train']:.4f}  val {losses['val']:.4f}  "
                  f"({elapsed:.0f}s)")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg, "tok_path": str(HERE/"bpe.json")},
                    HERE / "ckpt.pt",
                )

        # gradient accumulation: do `accum_steps` forward+backwards before stepping
        optim.zero_grad(set_to_none=True)
        for _ in range(hp["accum_steps"]):
            x, y = get_batch(train_data, hp["block_size"], hp["micro_batch"], hp["device"])
            _, loss = model(x, y)
            (loss / hp["accum_steps"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
        optim.step()

    print(f"done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
