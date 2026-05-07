import torch
import torch.nn as nn
def get_state_dict(model):
    return {name: p.data.clone() for name, p in model.named_parameters()}
def load_state_dict(model, sd):
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(sd[name])
if __name__ == "__main__":
    torch.manual_seed(0)
    m1 = nn.Linear(3, 2)
    sd = get_state_dict(m1)
    m2 = nn.Linear(3, 2)
    load_state_dict(m2, sd)
    eq = torch.equal(m1.weight, m2.weight) and torch.equal(m1.bias, m2.bias)
    print("ok" if eq else "FAIL")
