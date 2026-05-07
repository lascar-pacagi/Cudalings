import torch
def my_layernorm_backward(x, gamma, beta, dy, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, keepdim=True, unbiased=False)
    rstd = (var + eps).rsqrt()
    x_hat = (x - mean) * rstd
    a = dy * gamma
    a_mean = a.mean(dim=-1, keepdim=True)
    ax_mean = (a * x_hat).mean(dim=-1, keepdim=True)
    dx = rstd * (a - a_mean - x_hat * ax_mean)
    dgamma = (dy * x_hat).reshape(-1, x.shape[-1]).sum(dim=0)
    dbeta  = dy.reshape(-1, x.shape[-1]).sum(dim=0)
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
    ok = (torch.allclose(dx_mine, x.grad, atol=1e-5)
          and torch.allclose(dg_mine, gamma.grad, atol=1e-5)
          and torch.allclose(db_mine, beta.grad, atol=1e-5))
    print("ok" if ok else "FAIL")
