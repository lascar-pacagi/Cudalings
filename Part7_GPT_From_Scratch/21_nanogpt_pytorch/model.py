"""nanoGPT model in PyTorch -- a single, heavily-commented file.

This is a faithful reimplementation of the model from Karpathy's nanoGPT,
trimmed to the essentials and annotated with shape comments. Read it
top to bottom; every block builds on the previous one.

Tensor-shape conventions:
    B  = batch size                   T  = sequence length
    E  = n_embd                       H  = n_head
    Dh = head_dim = E // H            V  = vocab_size
"""

from dataclasses import dataclass

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GPTConfig:
    """Hyper-parameters for the model. Default = a tiny 1M-parameter GPT
    that trains in a few minutes on a Pascal GPU.

    Values are deliberately low — increase them for real experiments.
    """
    block_size: int = 256        # T: max context length
    vocab_size: int = 65         # V: char-level vocab for tinyshakespeare
    n_layer: int = 4             # number of transformer blocks
    n_head:  int = 4             # H: heads in each attention layer
    n_embd:  int = 128           # E: model width
    dropout: float = 0.0         # dropout prob; 0 disables (faster)
    bias:    bool  = False       # whether Linear/LN modules carry biases


# ---------------------------------------------------------------------------
# LayerNorm with optional bias.
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """Plain LayerNorm but with `bias=False` available — Karpathy's choice
    for nanoGPT to match the GPT-2 paper. Works identically to nn.LayerNorm
    when bias=True."""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.layer_norm handles the math: subtract mean, divide by std (over
        # the last dim), then scale + shift.
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


# ---------------------------------------------------------------------------
# Causal multi-head self-attention.
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """The heart of the transformer.

    We do all heads in a single matrix multiply (one big linear from E to 3E
    for q,k,v concatenated). Then we reshape to expose the H dim. Doing
    this as one matmul instead of three separate ones is ~2x faster --
    fewer kernel launches, better cuBLAS utilization.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head

        # one Linear projects x → [q, k, v] all at once
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # output projection back to E after the attention mix
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout  = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Causal mask -- a (1,1,T,T) lower-triangular ones matrix. Registered
        # as a buffer so .to(device) moves it but it doesn't show up as a
        # parameter to be optimized.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape

        # 1) project to qkv all at once
        qkv = self.c_attn(x)                      # (B, T, 3E)
        q, k, v = qkv.split(self.n_embd, dim=2)   # each: (B, T, E)

        # 2) reshape to (B, H, T, Dh) for per-head computation
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)   # (B, H, T, Dh)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3) scaled dot-product attention
        # att[B, H, t, s] = q[B,H,t,:] · k[B,H,s,:] / sqrt(Dh)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))   # (B,H,T,T)

        # 4) causal mask -- set future positions to -inf so softmax kills them
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 5) weighted sum of values
        y = att @ v                               # (B, H, T, Dh)

        # 6) merge heads back to (B, T, E) and project
        y = y.transpose(1, 2).contiguous().view(B, T, E)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ---------------------------------------------------------------------------
# MLP (FFN) block
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """The position-wise feed-forward block.
    Widens 4x, applies GELU, narrows back, dropout."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU "tanh approximation" matches OpenAI's GPT-2 weights and is
        # what we'll port to CUDA. The exact GELU uses an erf and is slower.
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


# ---------------------------------------------------------------------------
# Transformer block (norm + attn + residual; norm + mlp + residual)
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1  = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp  = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: norm BEFORE the sub-layer, residual carries x forward.
        # This is what GPT-2 does; lets you stack many blocks stably.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# The full GPT
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # token + positional embedding tables, plus final layer norm and head
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f = LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: the lm_head and wte share weights. Saves V*E params
        # and improves training stability (Press & Wolf 2017).
        self.transformer.wte.weight = self.lm_head.weight

        # init weights with the GPT-2 scheme: N(0, 0.02), output projections
        # get an extra 1/sqrt(2*n_layer) scaling.
        self.apply(self._init_weights)
        for pname, p in self.named_parameters():
            if pname.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) int64 token ids
        targets: (B, T) int64 next-token labels (or None for inference)
        Returns (logits, loss). Loss is None when targets is None.
        """
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence too long: {T} > {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)

        # forward the token + position embeddings
        tok_emb = self.transformer.wte(idx)        # (B, T, E)
        pos_emb = self.transformer.wpe(pos)        # (T, E)
        x = self.transformer.drop(tok_emb + pos_emb)

        # forward through all transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # training: compute loss over all positions
            logits = self.lm_head(x)               # (B, T, V)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss

        # inference: only the LAST position is needed for next-token prediction.
        # Slicing here saves the V-dim matmul on T-1 stale positions.
        logits = self.lm_head(x[:, [-1], :])       # (B, 1, V)
        return logits, None

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Greedy if temperature=0; multinomial otherwise.

        idx: (B, T_seed) initial context
        Returns: (B, T_seed + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # crop context so we never exceed block_size
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)   # (B, V)

            if top_k is not None:
                # zero out everything except the top_k logits
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            if temperature == 0.0:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
