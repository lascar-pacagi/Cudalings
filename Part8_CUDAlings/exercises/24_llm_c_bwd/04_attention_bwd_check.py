"""CUDAlings 24.04 -- Single-head attention backward, verified against autograd.

Forward (single head, no mask, no batch -- just T queries on T keys):
    att = softmax(q @ k.T / sqrt(Dh))
    out = att @ v

Backward decomposes into three matmul backwards + a softmax backward.
This exercise gives you PyTorch tensors and asks you to reproduce the
gradients without using torch.autograd's softmax shortcut.

Goal: compute (dq, dk, dv) yourself and match autograd's grads.
"""

# I AM NOT DONE

import math
import torch


def my_attention_backward(q, k, v, dout):
    """q, k, v shape: (T, Dh).  dout shape: (T, Dh).  Return (dq, dk, dv)."""
    T, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)

    # Forward (saving intermediates)
    att_logits = (q @ k.transpose(0, 1)) * scale
    att = torch.softmax(att_logits, dim=-1)

    # Backward (4 steps):
    # TODO: gradient through the att @ v matmul -> dv and datt
    # TODO: gradient through the row-wise softmax -> datt_logits (use the formula from 24.02)
    # TODO: gradient through the q @ k^T scaled-dot -> dq and dk (mind `scale`)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    return dq, dk, dv


if __name__ == "__main__":
    torch.manual_seed(0)
    T, Dh = 4, 8
    q = torch.randn(T, Dh, requires_grad=True)
    k = torch.randn(T, Dh, requires_grad=True)
    v = torch.randn(T, Dh, requires_grad=True)

    att_logits = (q @ k.transpose(0, 1)) / math.sqrt(Dh)
    att = torch.softmax(att_logits, dim=-1)
    out = att @ v
    dout = torch.randn_like(out)
    out.backward(dout)

    dq_m, dk_m, dv_m = my_attention_backward(q.detach(), k.detach(), v.detach(), dout)
    ok = (torch.allclose(dq_m, q.grad, atol=1e-5)
          and torch.allclose(dk_m, k.grad, atol=1e-5)
          and torch.allclose(dv_m, v.grad, atol=1e-5))
    print("ok" if ok else "FAIL")
