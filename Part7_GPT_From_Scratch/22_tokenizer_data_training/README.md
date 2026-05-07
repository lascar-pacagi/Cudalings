# Chapter 22 — Tokenizer, Data Pipeline, Chat-style Training

Chapter 21 trained a char-level GPT. Real LLMs use a **byte-pair encoding
(BPE)** tokenizer that compresses common substrings into single tokens —
"the" is one token, " transformer" might be one or two. This chapter:

1. Builds a minimal BPE tokenizer (~100 lines), and explains it from
   first principles. We also show how to drop in OpenAI's `tiktoken`
   for a real-vocab variant.
2. Wraps the tokenizer in a streaming dataloader that turns a raw text
   corpus into batched (B, T) tensors of token ids.
3. Adapts the training loop from chapter 21 to the BPE vocabulary, with
   warmup + cosine LR schedule, gradient accumulation, and best-checkpoint
   tracking — the production-realistic loop.
4. Adds an interactive **chat sampler** that takes user input, encodes
   it, generates a continuation, and prints the response — the Karpathy
   "minGPT can chat with you" experience.

The model itself is unchanged: `from model import GPT, GPTConfig` (we
import from chapter 21 — symlink or copy as you prefer).

## Why BPE matters for "chat"

A char-level model has to spell every word. A BPE model with V≈50000 has
words like " hello" or " machine" as single tokens. That's:
- ~4x fewer tokens per sentence → 4x more "context" in the same T
- Faster convergence (each gradient step covers more meaningful units)
- The base unit is a sub-word, so out-of-vocab words still encode (as
  pieces), unlike a word-level tokenizer.

For a "ChatGPT-like" experience at small scale, the tokenizer is what
makes chat conversations fit in a 256-token context window.

## Files

| File                  | What it does                                  |
|-----------------------|-----------------------------------------------|
| `tokenizer_bpe.py`    | Minimal BPE: train on a corpus, encode/decode |
| `data.py`             | Streaming dataloader; .bin shard format       |
| `train_chat.py`       | Training loop with LR schedule, accumulation  |
| `chat.py`             | Interactive REPL: type, get model response    |
| `prepare_dataset.py`  | One-shot script: download text → tokens.bin   |

## Quick start

```bash
cd Part7_GPT_From_Scratch/22_tokenizer_data_training

# 1. Train a BPE tokenizer + tokenize the dataset
python prepare_dataset.py     # downloads tinyshakespeare; trains BPE; writes tokens.bin

# 2. Train a 10M-param GPT
python train_chat.py          # 30 min on Quadro P4200 → ckpt.pt

# 3. Chat
python chat.py                # interactive sampling
```

## What you'll learn

- **BPE merges as a greedy compression algorithm** — each merge replaces
  the most-frequent adjacent pair until you've grown V tokens.
- **Token-id .bin shards** — the realistic data format for LLM training.
  Memory-mapped numpy arrays let you shuffle and slice multi-GB corpora
  with zero copy.
- **Cosine LR schedule + warmup** — what every modern LLM uses. Linear
  warmup for `warmup_iters`, then cosine decay to `min_lr`.
- **Gradient accumulation** — when your micro-batch doesn't fit in
  memory, accumulate gradients across N micro-steps before stepping.

The CUDAlings exercises in `Part8/exercises/22_tokenizer/` drill the BPE
encode/decode and the LR schedule; do those before writing your own
tokenizer from scratch.
