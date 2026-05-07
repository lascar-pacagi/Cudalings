import torch
def causal_mask(T):
    return torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
if __name__ == "__main__":
    m = causal_mask(4)
    print("ok" if (m[0,1] == float("-inf") and m[3,3] == 0.0 and m[1,0] == 0.0) else "FAIL")
