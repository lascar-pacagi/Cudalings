import torch
def im2col_then_matmul(x, W):
    IC, IH, IW = x.shape
    OC, _, KH, KW = W.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    cols = torch.zeros(IC * KH * KW, OH * OW)
    for kh in range(KH):
        for kw in range(KW):
            patch = x[:, kh:kh + OH, kw:kw + OW].reshape(IC, OH * OW)
            row = (kh * KW + kw)
            cols[row::KH * KW] = patch  # rows for each ic at offset (kh*KW + kw)
    # Easier: use torch.unfold
    cols = torch.nn.functional.unfold(x.unsqueeze(0), (KH, KW)).squeeze(0)
    Wm = W.reshape(OC, IC * KH * KW)
    y = (Wm @ cols).reshape(OC, OH, OW)
    return y
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(3, 8, 8)
    W = torch.randn(4, 3, 3, 3)
    mine = im2col_then_matmul(x, W)
    ref = torch.nn.functional.conv2d(x.unsqueeze(0), W).squeeze(0)
    print("ok" if torch.allclose(mine, ref, atol=1e-5) else f"FAIL {(mine - ref).abs().max()}")
