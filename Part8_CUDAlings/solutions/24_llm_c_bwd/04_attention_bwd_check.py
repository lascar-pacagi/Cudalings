import math
import torch
def my_attention_backward(q, k, v, dout):
    T, Dh = q.shape
    scale = 1.0 / math.sqrt(Dh)
    att_logits = (q @ k.transpose(0, 1)) * scale
    att = torch.softmax(att_logits, dim=-1)
    dv = att.transpose(0, 1) @ dout
    datt = dout @ v.transpose(0, 1)
    sum_term = (att * datt).sum(dim=-1, keepdim=True)
    datt_logits = att * (datt - sum_term)
    dq = (datt_logits @ k) * scale
    dk = (datt_logits.transpose(0, 1) @ q) * scale
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
