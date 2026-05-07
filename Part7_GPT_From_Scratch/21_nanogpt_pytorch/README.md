# Chapter 21 — nanoGPT in PyTorch

A complete, ~300-line implementation of a decoder-only Transformer
(GPT-style) in PyTorch. We trace every tensor shape and every gradient
path so when we port it to CUDA in chapter 23, the translation is line
by line.

## Table of Contents
1. [What is a GPT?](#what-is-a-gpt)
2. [Tensor Shape Conventions](#tensor-shape-conventions)
3. [Token + Positional Embeddings](#token--positional-embeddings)
4. [LayerNorm](#layernorm)
5. [Causal Multi-Head Self-Attention](#causal-multi-head-self-attention)
6. [The MLP (FFN) Block](#the-mlp-ffn-block)
7. [Stacking Blocks: A Transformer](#stacking-blocks-a-transformer)
8. [The Language-Modeling Head + Loss](#the-language-modeling-head--loss)
9. [Sampling / Generation](#sampling--generation)
10. [Programs in This Chapter](#programs-in-this-chapter)

---

## What is a GPT?

```
                              GPT in one diagram
                              ==================

   tokens:   [The, cat, sat, on, the, mat]                   shape: (B, T)
                          │
                          ▼
   wte:  one row of E per token        + wpe: position 0..T-1
                          │
                          ▼
                 sum  →  shape (B, T, E)
                          │
                          ▼
                 ┌────────────────────────┐
                 │    Block 1             │
                 │  ┌──────────────────┐  │
                 │  │ LayerNorm        │  │   "pre-norm" GPT-2 style
                 │  │ Causal MH-Attn   │  │
                 │  │ + residual       │  │
                 │  ├──────────────────┤  │
                 │  │ LayerNorm        │  │
                 │  │ MLP (4x widen)   │  │
                 │  │ + residual       │  │
                 │  └──────────────────┘  │
                 └──────────┬─────────────┘
                            ▼
                          ... × N_LAYER blocks ...
                            ▼
                       LayerNorm
                            ▼
              Linear (E → vocab_size)              "lm_head"
                            ▼
                      logits  shape (B, T, V)
                            ▼
                  cross-entropy vs targets[t+1]
```

A GPT is just **N stacked transformer blocks**, where each block is
**(norm → attention → +) → (norm → MLP → +)**. The output is mapped back
to vocabulary logits and trained to predict the next token. That's it.

Every other detail is implementation. Once you understand this picture,
the only question is: how do we make each piece fast on a GPU?

## Tensor Shape Conventions

The code uses these symbol names throughout:

| symbol  | meaning                                      | example |
|---------|----------------------------------------------|---------|
| `B`     | batch size                                   | 32      |
| `T`     | sequence length (block_size)                 | 256     |
| `E`     | embedding dim (n_embd)                       | 384     |
| `H`     | number of attention heads (n_head)           | 6       |
| `Dh`    | per-head dim = E / H                         | 64      |
| `V`     | vocab size                                   | 65 (chars) or ~50000 (BPE) |

All tensors are written shape-annotated in the comments: `# (B, T, E)`.
This is the single most important habit when working with attention —
shape bugs are silent and mistraining runs are expensive.

## Token + Positional Embeddings

```
   tokens     :  (B, T)        ints in [0, V)
   wte        :  (V, E)        one row per token in the vocab
   wte[tokens]:  (B, T, E)     gathered token vectors

   positions  :  arange(T)  =  (T,)   ints in [0, T)
   wpe        :  (T, E)        one row per position
   wpe[pos]   :  (T, E)
   wpe[pos][None] : (1, T, E)  broadcasts across the batch

   x = wte[tokens] + wpe[pos]   # (B, T, E)
```

GPT-2 uses **learned** positional embeddings — just another lookup table.
GPT-NeoX/Llama use rotary; we'll stick with learned for fidelity to
Karpathy's nanoGPT.

## LayerNorm

LayerNorm normalizes each token's E-dim vector to have mean 0, variance 1,
then re-scales with learned `gamma` and `beta`:

```
   for each token x ∈ R^E:
     μ  = mean(x)             scalar
     σ² = var(x)              scalar
     x̂ = (x - μ) / sqrt(σ² + ε)
     y  = γ ⊙ x̂ + β         γ, β ∈ R^E learned
```

Pre-norm GPT-2 puts LN at the **start** of each sub-block (before attention,
before MLP). This stabilizes gradients in deep stacks and is what every
modern LLM does.

## Causal Multi-Head Self-Attention

The hardest piece. Here's the full math:

```
  Input x: (B, T, E)

  1. Linear projection:
     qkv = x · W_qkv         W_qkv ∈ R^{E × 3E}
     qkv shape: (B, T, 3E) → split into q, k, v each (B, T, E)

  2. Reshape to per-head:
     q : (B, T, H, Dh) → (B, H, T, Dh)
     k : (B, T, H, Dh) → (B, H, T, Dh)
     v : (B, T, H, Dh) → (B, H, T, Dh)

  3. Scaled dot-product:
     attn_logits = q @ k.transpose(-2, -1) / sqrt(Dh)   shape: (B, H, T, T)

  4. Causal mask:
     attn_logits[..., t, s] = -inf for s > t           # don't peek into the future
     attn_weights = softmax(attn_logits, dim=-1)

  5. Weighted sum of values:
     y = attn_weights @ v        shape: (B, H, T, Dh)

  6. Merge heads + output projection:
     y : (B, H, T, Dh) → (B, T, H, Dh) → (B, T, E)
     out = y · W_out             W_out ∈ R^{E × E}
```

Why **causal**? At training time, each token gets to attend only to itself
and the tokens before it. That's how a language model learns to predict
the next token without cheating: it never sees the answer in its own
input. The mask is what makes this *autoregressive*.

Why **multi-head**? With H heads, the model can specialize: one head
might track noun→adjective links, another long-range coreference, etc.
Splitting E into H groups of Dh is essentially "free" parameter-wise
(same total weights), but adds a useful inductive bias.

## The MLP (FFN) Block

```
   x : (B, T, E)
   h = gelu(x · W1 + b1)        W1 ∈ R^{E × 4E}, h ∈ R^{B,T,4E}
   y = h · W2 + b2              W2 ∈ R^{4E × E}, y ∈ R^{B,T,E}
```

The "4x widen" is GPT convention: hidden dim of MLP is 4×E. GELU is the
activation: `0.5 * x * (1 + tanh(...))`. (We'll see the closed form when
we port it to CUDA — there's a tanh approximation that's much faster.)

## Stacking Blocks: A Transformer

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

Three lines. That's it. The rest of GPT is bookkeeping.

## The Language-Modeling Head + Loss

After all blocks + a final LayerNorm, we map `(B, T, E) → (B, T, V)`
with a single Linear (often *tied* to wte: same matrix used for both
embedding lookup and output projection — saves V×E parameters).

The loss is plain cross-entropy of logits[t] against tokens[t+1], summed
over the sequence and averaged over the batch.

## Sampling / Generation

To generate text:

```python
def generate(model, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]            # crop context
        logits, _ = model(idx_cond)                # (B, T, V)
        logits = logits[:, -1, :] / temperature    # take last position only
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

That's the entire generation loop. Greedy decoding uses `argmax` instead
of `multinomial`; top-k filters logits before softmax. We implement both.

---

## Programs in This Chapter

| File              | What it does                                      |
|-------------------|---------------------------------------------------|
| `model.py`        | The GPT model: blocks, attention, MLP             |
| `train.py`        | Training loop on Tiny Shakespeare, char-level     |
| `sample.py`       | Load a checkpoint and generate text               |
| `requirements.txt`| `torch`, `numpy`                                  |

```bash
cd Part7_GPT_From_Scratch/21_nanogpt_pytorch
pip install -r requirements.txt
python train.py     # ~5 min on Quadro P4200; saves ckpt.pt
python sample.py    # generates 500 chars of (very approximate) Shakespeare
```

The CUDAlings exercises in `Part8/exercises/21_nanogpt_py/` drill the
attention math piece by piece — start there if you want to internalize
the shapes before reading `model.py`.
