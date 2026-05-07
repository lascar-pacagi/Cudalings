# Part 7 — Build a ChatGPT-like Assistant from Scratch

> **The bridge between "I can write CUDA kernels" and "I built my own
> transformer-based assistant."** First in PyTorch (so you see attention
> work), then ported into pure CUDA C++ (so you see what makes it fast).

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                     PART 7 ROADMAP                                       │
 │                                                                          │
 │  Ch 21: nanoGPT in PyTorch          Ch 22: tokenizer + data + training   │
 │  ┌────────────────────────────┐    ┌────────────────────────────────┐    │
 │  │ token+pos embeddings       │    │ char tokenizer + BPE           │    │
 │  │ multi-head causal attn     │───►│ tinyshakespeare loader         │    │
 │  │ MLP block, layernorm       │    │ training loop + AdamW + eval   │    │
 │  │ generation / sampling      │    │ checkpoints + chat sampling    │    │
 │  └────────────────────────────┘    └────────────────────────────────┘    │
 │                                                  │                       │
 │                                                  ▼                       │
 │  Ch 23: llm.c forward in CUDA       Ch 24: llm.c backward                │
 │  ┌────────────────────────────┐    ┌────────────────────────────────┐    │
 │  │ encoder, layernorm, gelu   │    │ matmul_bwd, ln_bwd, attn_bwd   │    │
 │  │ matmul (fwd), softmax_attn │───►│ encoder_bwd, gelu_bwd          │    │
 │  │ residual, head             │    │ verify against PyTorch grads   │    │
 │  └────────────────────────────┘    └────────────────────────────────┘    │
 │                                                  │                       │
 │                                                  ▼                       │
 │  Ch 25: llm.c full training in pure CUDA C++                             │
 │  ┌────────────────────────────────────────────────────────────────┐      │
 │  │ AdamW kernel, weight init, full forward/backward loop          │      │
 │  │ trains the same nanoGPT, sampled output is comparable          │      │
 │  └────────────────────────────────────────────────────────────────┘      │
 │                                                                          │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Why PyTorch first, CUDA second?

Karpathy's `nanoGPT` is the cleanest reference implementation of a GPT in
the world: ~300 lines of PyTorch, no abstractions you don't want. We start
there so you see the math working before you rewrite it on the metal.

Then `llm.c` (Karpathy's pure-C/CUDA port) is the same model, but every
operation is a kernel you call directly. Knowing what the PyTorch ops do
makes the CUDA versions read like a translation, not a brand-new program.

By the end you'll have **both**: a PyTorch model that trains and chats,
and a CUDA-only program that produces identical outputs (within rounding).
That's the same exercise that turned Karpathy into the best teacher of
modern deep learning, and it'll work for you.

## Pascal-specific notes

The Quadro P4200 (sm_61) has:
- **No tensor cores** — wmma instructions don't exist. matmul stays as
  `fma` ops on FP32. We won't use TF32 or bf16.
- **No `cp.async`** — Pascal can't do async global→shared copies. Tiled
  loads are synchronous.
- **No flash-attention kernel** — even simple online-softmax attention
  needs `__shfl_sync`, which Pascal supports (sm_30+), so we're OK there.

The implementation in chapter 23 sticks to features that work on Pascal.
A note marks each kernel that would benefit from newer hardware so you
can swap it out when you upgrade.

## Models you'll train

| Chapter | Tokens | Params | Train time on P4200 (approx) |
|---------|--------|--------|-------------------------------|
| 21      | 1.1M (Tiny Shakespeare, char-level) | ~1M  | 5 min      |
| 22      | 1.1M (Tiny Shakespeare, BPE)         | ~10M | 30 min     |
| 25      | 1.1M, same model in CUDA             | ~10M | 60-90 min  |

Don't expect ChatGPT-quality output — at this scale, the model produces
plausible Shakespeare-style nonsense. The point is the **mechanism**: you
end with a complete inference + training stack you understand.

## Per-chapter index

- `21_nanogpt_pytorch/` — single-file PyTorch GPT, with annotated math.
- `22_tokenizer_data_training/` — BPE, dataloader, training loop, sampling.
- `23_llm_c_forward/` — every forward kernel as standalone CUDA, plus a
  driver that runs them all and matches PyTorch's outputs.
- `24_llm_c_backward/` — backward kernels and gradient verification.
- `25_llm_c_training/` — AdamW kernel and the full training loop in CUDA.

Each chapter has the standard layout: `README.md` (theory), `Makefile` /
`requirements.txt`, source files. The CUDAlings exercises in `Part8/`
chapters 21-25 drill the same code you'll see here.
