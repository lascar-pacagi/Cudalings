"""CUDAlings 24.03 -- Layernorm backward, verified against PyTorch autograd.

Forward (per row of E elements):
    mean = x.mean()
    var  = x.var(unbiased=False)
    rstd = 1 / sqrt(var + eps)
    x_hat = (x - mean) * rstd
    y = gamma * x_hat + beta

Backward (compact form, llm.c style):
    dgamma = sum(dy * x_hat, dim=0)
    dbeta  = sum(dy, dim=0)
    a = (dy * gamma)
    dx = rstd * (a - a.mean(dim=-1, keepdim=True) - x_hat * (a * x_hat).mean(dim=-1, keepdim=True))

Goal: implement my_layernorm_backward and verify against PyTorch's autograd
on a (B, T, E) input with random data.
"""

# I AM NOT DONE

import torch


def my_layernorm_backward(x, gamma, beta, dy, eps=1e-5):
    """Return (dx, dgamma, dbeta)."""
    # Forward intermediates
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)
    rstd = (var + eps).rsqrt()
    x_hat = (x - mean) * rstd

    # TODO: implement the compact layernorm backward from the docstring:
    #       both `dx` (uses two row-mean reductions of dy*gamma and dy*gamma*x_hat)
    #       and the parameter grads `dgamma`, `dbeta` (sums over batch dims).
    dx = torch.zeros_like(x)
    dgamma = torch.zeros_like(gamma)
    dbeta  = torch.zeros_like(beta)
    return dx, dgamma, dbeta


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, E = 2, 3, 8
    x = torch.randn(B, T, E, requires_grad=True)
    gamma = torch.randn(E, requires_grad=True)
    beta  = torch.randn(E, requires_grad=True)

    y_ref = torch.nn.functional.layer_norm(x, (E,), gamma, beta)
    dy = torch.randn_like(y_ref)
    y_ref.backward(dy)

    dx_mine, dg_mine, db_mine = my_layernorm_backward(x.detach(), gamma.detach(),
                                                     beta.detach(), dy)

    ok = (torch.allclose(dx_mine, x.grad,    atol=1e-5)
          and torch.allclose(dg_mine, gamma.grad, atol=1e-5)
          and torch.allclose(db_mine, beta.grad,  atol=1e-5))
    print("ok" if ok else "FAIL")
