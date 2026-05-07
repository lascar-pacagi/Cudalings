# Chapter 24 — llm.c-style backward pass

Every forward kernel in chapter 23 has a backward partner. The pattern is
the same as in Part 4 of the course: read `dy` (gradient of the output)
plus any saved tensors, write `dx` (gradient of the input) and optionally
`dW` (gradient of the parameters). Then sum gradients across the batch
dimension into the parameter buffers.

## Per-op gradient cheatsheet

```
   matmul: y = x · W^T (+ b)
     dx = dy · W
     dW = dy^T · x          (note transposed sum)
     db = sum(dy, axis=0)

   layernorm: y = γ · (x - μ) / σ + β
     dγ = sum(dy · (x - μ)/σ, axis=batch)
     dβ = sum(dy, axis=batch)
     dx = γ/σ · (dy - mean(dy * γ) - x_hat * mean(dy * γ * x_hat))
       (Karpathy's compact form; see llm.c/layernorm_backward)

   gelu:    y = 0.5x(1 + tanh(k(x + cx^3)))
     local Jacobian; closed form, see kernel.

   softmax: p = exp(x_i - m) / sum exp
     dx_i = p_i * (dy_i - sum_j p_j dy_j)

   attention:    composes the three above in the right order.

   residual: y = x1 + x2  →  dx1 = dy, dx2 = dy

   encoder: y = wte[idx] + wpe[t]
     scatter-add: dwte[idx] += dy ; dwpe[t] += dy
     (use atomicAdd because multiple positions can hit the same token id)

   embedding tying: lm_head shares wte. The gradient through lm_head's
   matmul writes into dwte (same buffer as the embedding gradient);
   atomicAdd keeps both contributions safe.
```

## Files you'll write

```
gpt2_backward.cu          # all backward kernels + driver
verify_grad.py            # numerical gradient check vs PyTorch's autograd
```

The skeleton in this directory has each kernel signature + a TODO body.
The CUDAlings exercises in `Part8/exercises/24_llm_c_bwd/` walk you
through one backward at a time, gradient-checking each before stacking
them in the driver.

## Numerical gradient check

```python
# verify_grad.py
y_torch, dx_torch = pytorch_forward_backward(x, ...)
y_cuda            = run_cuda_forward(x, ...)
dx_cuda           = run_cuda_backward(...)
print((dx_torch - torch.from_numpy(dx_cuda)).abs().max())  # < 1e-3 = pass
```

Do this for one kernel at a time. If `matmul_backward` matches but the
full block doesn't, the bug is one or two layers up — easy to bisect.

## What this chapter teaches

- **Atomic adds** for gradient accumulation when multiple sources write
  to the same parameter slot (every layer's lm_head row update).
- **Compact layernorm backward** — the four-line form most CUDA libs
  use, derived from chain rule.
- **Backwards through softmax + matmul** as a fused 3-step kernel
  (attention backward) — this is where most LLM training time is spent.

After this chapter you have everything needed to **train** in CUDA, no
PyTorch involved. Chapter 25 wires it together.
