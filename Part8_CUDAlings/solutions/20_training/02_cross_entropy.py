import torch
def xent(logits, targets):
    m = logits.max(dim=-1, keepdim=True).values
    logsumexp = (logits - m).exp().sum(dim=-1, keepdim=True).log() + m
    log_probs = logits - logsumexp
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()
if __name__ == "__main__":
    torch.manual_seed(0)
    logits  = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    mine = xent(logits, targets)
    ref  = torch.nn.functional.cross_entropy(logits, targets)
    print("ok" if torch.allclose(mine, ref, atol=1e-5) else f"FAIL {mine} vs {ref}")
