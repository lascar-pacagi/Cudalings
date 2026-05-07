# Chapter 23 — llm.c-Style Forward Pass in Pure CUDA

We now translate every PyTorch op from chapter 21 into a hand-written
CUDA kernel. The structure follows Karpathy's
[llm.c](https://github.com/karpathy/llm.c) — one kernel per primitive,
called from a top-level `gpt2_forward` driver. No PyTorch, no cuBLAS, no
external libraries beyond `cuda_runtime`.

```
   PyTorch op                        CUDA kernel (this chapter)
   ──────────────                    ──────────────────────────
   wte[idx] + wpe[pos]      ───►     encoder_forward
   F.layer_norm             ───►     layernorm_forward
   x @ W                    ───►     matmul_forward     (naive then tiled)
   q @ k.T / sqrt(Dh) +mask ───►     attention_forward  (online softmax)
   F.softmax(..., dim=-1)            (subsumed into attention_forward)
   @ V                               (subsumed into attention_forward)
   F.gelu(x, "tanh")        ───►     gelu_forward
   x + y                    ───►     residual_forward
   final lm_head            ───►     matmul_forward (reused)
```

## Tensor layout convention

We use the same shapes as PyTorch (B, T, E etc.) but flat 1D buffers,
because there's no `Tensor` class — just `float*` pointers.
`x[b, t, e]` lives at offset `b*T*E + t*E + e`.

```
   x:  (B, T, E)  → float* of size B*T*E
   q:  (B, H, T, Dh)  but stored as (B, T, H*Dh) = (B, T, E)
                       and re-indexed in attention
```

## What you'll build, kernel by kernel

| Kernel                  | Math                                             | File                |
|-------------------------|--------------------------------------------------|---------------------|
| `encoder_forward`       | `out[b,t,e] = wte[idx[b,t], e] + wpe[t, e]`     | `encoder.cu`        |
| `layernorm_forward`     | per-token normalize + γ scale + β shift          | `layernorm.cu`      |
| `matmul_forward`        | `out = x @ W^T (+ bias)`                         | `matmul.cu`         |
| `attention_forward`     | `q @ k.T / sqrt(Dh) + mask, softmax, @ v`        | `attention.cu`      |
| `gelu_forward`          | `0.5 * x * (1 + tanh(...))`                      | `gelu.cu`           |
| `residual_forward`      | `out = x + y` elementwise                        | `residual.cu`       |
| `softmax_forward`       | numerically-stable row-wise softmax              | `softmax.cu`        |

The driver `gpt2_forward.cu` chains them in the same order as
`Block.forward()` from chapter 21:

```
encoder_forward(x, idx, wte, wpe, B, T, E)
for layer in 0..n_layer:
    layernorm_forward(ln1_out, x, ln1_w, ln1_b)
    matmul_forward(qkv, ln1_out, qkv_w, qkv_b)        # x → q,k,v packed
    attention_forward(att_out, qkv, B, T, E, n_head)  # the heavy lifter
    matmul_forward(attn_proj_out, att_out, proj_w, proj_b)
    residual_forward(x, attn_proj_out)                # in-place add
    layernorm_forward(ln2_out, x, ln2_w, ln2_b)
    matmul_forward(fc_out, ln2_out, fc_w, fc_b)       # E → 4E
    gelu_forward(fc_out)                              # in-place
    matmul_forward(mlp_out, fc_out, fc2_w, fc2_b)     # 4E → E
    residual_forward(x, mlp_out)
layernorm_forward(x_final, x, ln_f_w, ln_f_b)
matmul_forward(logits, x_final, lm_head_w)             # last linear
```

## How we verify correctness

`verify.py` is a Python script that:
1. Builds a small GPT in PyTorch (chapter 21 `model.py`).
2. Saves all weights + a fixed input batch as raw `.bin` files.
3. Runs the same forward pass in CUDA C++.
4. Loads the CUDA outputs and compares against PyTorch's.

If max-abs-diff per intermediate < 1e-3, you've got a correct port.
Otherwise the script tells you which kernel diverged first.

## Build + run

```bash
cd Part7_GPT_From_Scratch/23_llm_c_forward
make             # builds gpt2_forward + every per-kernel test
make test        # runs all kernel tests in isolation, then end-to-end

# Compare against PyTorch:
python verify.py  # downloads weights from chapter 21 ckpt, runs both, diffs.
```

## Pascal vs newer GPU notes

- `attention.cu` uses online-softmax-style accumulation but does NOT use
  flash-attention's ping-pong tiling (which needs cp.async). On Pascal
  this is fine; on Volta+ you can swap in flash-attn for a ~2x speedup.
- `matmul.cu` is a tiled matmul without tensor cores. On any sm_70+ GPU
  you'd add wmma intrinsics for FP16/TF32.
- All our kernels are FP32. No mixed precision on Pascal (no Tensor Cores).

The point of this chapter is **correctness**: every gradient and every
forward output should match PyTorch. Speed comes from chapter 25's
optimizations + your hardware.
