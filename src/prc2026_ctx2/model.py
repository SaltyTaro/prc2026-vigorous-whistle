"""Context network, round 2: the round-1 architecture (prc2026_ctx.model.ContextNet) with optional self-attention layers among
the neighbour tokens before pooling (pre-norm transformer encoder layers, padding masked), so that the pooled context can depend on
how neighbours relate to each other (queue order, pushback sequence) and not only on each neighbour's relation to the focal row.
With `sa_layers = 0` the forward pass is the round-1 network's.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from prc2026_ctx.model import N_WAKE, TOKEN_FEATURES
from prc2026_nn.model import NumericEncoder, embedding_dim


class ContextNet2(nn.Module):
    def __init__(self, edges: np.ndarray, n_bins: np.ndarray, cardinalities: list[int], *, d_num: int = 8, cat_cap: int = 16, width: int = 512,
                 depth: int = 3, dropout: float = 0.1, tok_dim: int = 96, heads: int = 4, max_tokens: int = 128, wake_dim: int = 4,
                 sa_layers: int = 0, sa_heads: int = 4, sa_ff: int = 2, sa_dropout: float = 0.0):
        super().__init__()
        self.numeric = NumericEncoder(edges, n_bins, d_num)
        self.embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim(c, cat_cap)) for c in cardinalities])
        d_in = self.numeric.out_features + edges.shape[0] + sum(e.embedding_dim for e in self.embeddings)
        f = len(TOKEN_FEATURES)
        self.wake = nn.Embedding(N_WAKE, wake_dim)
        self.token = nn.Sequential(nn.Linear(2 * f + wake_dim, tok_dim), nn.ReLU(), nn.Linear(tok_dim, tok_dim), nn.ReLU())
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(tok_dim, sa_heads, sa_ff * tok_dim, sa_dropout, batch_first=True, norm_first=True)
                                      for _ in range(sa_layers)])
        if tok_dim % heads:
            raise ValueError("tok_dim must be a multiple of heads.")
        self.heads, self.dk = heads, tok_dim // heads
        self.query = nn.Linear(d_in, tok_dim)
        self.key = nn.Linear(tok_dim, tok_dim)
        self.value = nn.Linear(tok_dim, tok_dim)
        self.norm_attn = nn.LayerNorm(tok_dim)
        self.norm_mean = nn.LayerNorm(tok_dim)
        self.max_tokens = max_tokens
        layers, previous = [], d_in + 2 * tok_dim + 1
        for _ in range(depth):
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 1)
        nn.init.uniform_(self.head.weight, -1.0 / math.sqrt(previous), 1.0 / math.sqrt(previous))
        nn.init.zeros_(self.head.bias)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, tok: torch.Tensor, tok_nan: torch.Tensor, tok_wake: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        h, nan = self.numeric(x_num)
        z = torch.cat([h, nan.float()] + [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)], dim=1)
        e = self.token(torch.cat([tok, tok_nan, self.wake(tok_wake)], dim=-1))
        if len(self.encoder):
            # rows without any valid token keep one unmasked (zeroed) slot so attention stays defined; the pooling below masks it again
            none = ~valid.any(dim=1)
            pad = ~valid
            if none.any():
                pad = pad.clone()
                pad[none, 0] = False
            for layer in self.encoder:
                e = layer(e, src_key_padding_mask=pad)
            e = e * valid.unsqueeze(-1)
        b, k, d = e.shape
        mask = valid.float()
        q = self.query(z).view(b, self.heads, self.dk)
        keys = self.key(e).view(b, k, self.heads, self.dk)
        vals = self.value(e).view(b, k, self.heads, self.dk)
        scores = torch.einsum("bhd,bkhd->bhk", q, keys) / math.sqrt(self.dk)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1) * mask.unsqueeze(1)
        attn = torch.einsum("bhk,bkhd->bhd", weights, vals).reshape(b, d)
        count = mask.sum(dim=1, keepdim=True)
        mean = (e * mask.unsqueeze(-1)).sum(dim=1) / count.clamp(min=1.0)
        pooled = torch.cat([self.norm_attn(attn), self.norm_mean(mean), count / self.max_tokens], dim=1)
        return self.head(self.backbone(torch.cat([z, pooled], dim=1)))
