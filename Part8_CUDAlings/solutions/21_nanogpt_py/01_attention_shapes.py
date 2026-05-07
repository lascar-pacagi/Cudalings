import torch
def reshape_to_heads(x, n_head):
    B, T, E = x.shape
    Dh = E // n_head
    return x.view(B, T, n_head, Dh).transpose(1, 2)
if __name__ == "__main__":
    x = torch.arange(2 * 3 * 8, dtype=torch.float32).view(2, 3, 8)
    y = reshape_to_heads(x, 2)
    if y.shape == (2, 2, 3, 4) and y[0, 0, 0, 0] == 0 and y[0, 1, 0, 0] == 4:
        print("ok")
    else:
        print("FAIL")
