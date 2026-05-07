# Chapter 25 — Full Training Loop in Pure CUDA

The capstone. With chapter 23's forward and chapter 24's backward in
place, all that's left is:

1. **AdamW kernel** — the optimizer.
2. **A glue program** that loops:
   - `gpt2_forward(...)`
   - `crossentropy_softmax_backward(...)`
   - `gpt2_backward(...)`
   - `adamw_step(...)` for every parameter buffer
3. **Sampling** in CUDA: argmax over the last logit row, append, repeat.

```
   Per training step:
   ────────────────────────────────────
       sample_batch (host)        ── draws (B, T) tokens from train.bin
              │
              ▼
       cudaMemcpy host → d_idx
              │
              ▼
       gpt2_forward(...)           ── chapter 23 kernels
              │
              ▼
       softmax + xent (kernel)    ── one fused kernel; dlogits is the seed
              │
              ▼
       gpt2_backward(...)          ── chapter 24 kernels, fills dW for all
              │
              ▼
       adamw_step(...)             ── this chapter
              │
              └── back to top with new tokens
```

## AdamW kernel

For each parameter `p` with gradient `g`:

```
    m  = β1 * m + (1 - β1) * g
    v  = β2 * v + (1 - β2) * g²
    m̂ = m / (1 - β1^t)
    v̂ = v / (1 - β2^t)
    p ← p - lr * (m̂ / (sqrt(v̂) + eps) + wd * p)         # decoupled wd = "AdamW"
```

Trivially parallel: one thread per parameter element. We pass each
parameter buffer + its `m, v` state to the same kernel; the dispatcher
loops over parameter buffers.

## Files (skeleton)

| File                | What it does                                       |
|---------------------|----------------------------------------------------|
| `adamw.cu`          | The AdamW kernel + per-buffer dispatcher           |
| `train_gpt2.cu`     | The full training program                          |
| `Makefile`          | Builds against gpt2_forward / gpt2_backward        |
| `compare_pytorch.py`| Trains the same model in PyTorch, diffs the loss   |

## Verification: train both, expect the same loss curves

The most satisfying test in the course: train the chapter 21 PyTorch
model and the chapter 25 CUDA model from the SAME initial weights with
the SAME data ordering. Their loss curves should overlap to ~3
significant figures over 1000 steps. Any drift means a bug in your
backward.

```bash
python compare_pytorch.py        # writes pytorch_loss.csv
./train_gpt2 --steps 1000        # writes cuda_loss.csv
python plot_diff.py              # overlays the two and prints max-diff
```

## What you'll have built by the end

- ~1500 lines of pure CUDA C++ (no PyTorch, no cuBLAS).
- A model that trains on Tiny Shakespeare, samples Shakespeare-like text,
  and matches PyTorch's loss curve.
- A complete mental model of every kernel in a transformer.

This is the same program Karpathy distilled into `llm.c` -- you've
written your own version, on your own GPU, from first principles.
You can now read any LLM training codebase (Megatron, FSDP, vLLM) and
recognize the kernels. **You have built your own pytorch.**
