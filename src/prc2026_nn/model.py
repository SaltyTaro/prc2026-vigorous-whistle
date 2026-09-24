"""The network: piecewise-linear numeric embeddings, missing-value flags, categorical embeddings and a ReLU MLP.

`k > 1` turns the MLP into a TabM-mini ensemble (Gorishniy et al., 2024): k multiplicative input adapters (random +-1
signs) in front of one shared backbone, k linear heads; every member is trained on the full loss and the prediction is
the member mean. Numeric encoding follows the piecewise-linear "Q-LR" embedding (Gorishniy et al., 2022): each feature is
encoded against its fitting-row quantile edges, mapped by its own linear layer and passed through a ReLU. With
`numeric_mode="rank"` each feature enters as one scalar, its fractional quantile rank minus one half (no embedding).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def embedding_dim(cardinality: int, cap: int) -> int:
    return int(min(cap, max(2, round(1.6 * cardinality ** 0.56))))


class NumericEncoder(nn.Module):
    def __init__(self, edges: np.ndarray, n_bins: np.ndarray, d: int):
        super().__init__()
        f, bins = edges.shape[0], edges.shape[1] - 1
        width = edges[:, 1:] - edges[:, :-1]
        valid = np.arange(bins)[None, :] < n_bins[:, None]
        width = np.where(valid & (width > 0), width, 1.0)
        self.register_buffer("left", torch.tensor(edges[:, :-1], dtype=torch.float32))
        self.register_buffer("inv_width", torch.tensor(1.0 / width, dtype=torch.float32))
        self.register_buffer("valid", torch.tensor(valid, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(n_bins, dtype=torch.float32))
        self.d = d
        if d > 0:
            self.weight = nn.Parameter(torch.empty(f, bins, d))
            self.bias = nn.Parameter(torch.empty(f, d))
            bound = 1.0 / math.sqrt(bins)
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.bias, -bound, bound)
        self.out_features = f * d if d > 0 else f

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, F) raw values -> (N, F, bins) piecewise-linear codes in [0, 1] (all zero where the value is missing), and the NaN mask."""
        nan = torch.isnan(x)
        codes = ((torch.nan_to_num(x, nan=0.0).unsqueeze(-1) - self.left) * self.inv_width).clamp(0.0, 1.0) * self.valid
        return codes.masked_fill(nan.unsqueeze(-1), 0.0), nan

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        codes, nan = self.encode(x)
        if self.d > 0:
            h = torch.relu(torch.einsum("nfb,fbd->nfd", codes, self.weight) + self.bias).flatten(1)
        else:
            h = (codes.sum(-1) / self.count - 0.5).masked_fill(nan, 0.0)
        return h, nan


class TabularNet(nn.Module):
    def __init__(self, edges: np.ndarray, n_bins: np.ndarray, cardinalities: list[int], *, numeric_mode: str = "ple", d_num: int = 8,
                 cat_cap: int = 16, width: int = 512, depth: int = 3, dropout: float = 0.1, k: int = 1):
        super().__init__()
        if numeric_mode not in ("ple", "rank"):
            raise ValueError(f"Unknown numeric mode {numeric_mode}.")
        self.numeric = NumericEncoder(edges, n_bins, d_num if numeric_mode == "ple" else 0)
        self.embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim(c, cat_cap)) for c in cardinalities])
        d_in = self.numeric.out_features + edges.shape[0] + sum(e.embedding_dim for e in self.embeddings)
        self.k = k
        if k > 1:
            signs = torch.randint(0, 2, (k, d_in)).float() * 2 - 1
            self.adapter = nn.Parameter(signs)
        layers, previous = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        self.backbone = nn.Sequential(*layers)
        self.head_weight = nn.Parameter(torch.empty(k, previous))
        self.head_bias = nn.Parameter(torch.zeros(k))
        nn.init.uniform_(self.head_weight, -1.0 / math.sqrt(previous), 1.0 / math.sqrt(previous))
        self.d_in = d_in

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """(N, k) member predictions in the scaled target unit."""
        h, nan = self.numeric(x_num)
        parts = [h, nan.float()] + [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
        z = torch.cat(parts, dim=1)
        n = z.shape[0]
        if self.k > 1:
            z = (z.unsqueeze(1) * self.adapter).reshape(n * self.k, self.d_in)
        z = self.backbone(z).reshape(n, self.k, -1)
        return torch.einsum("nkw,kw->nk", z, self.head_weight) + self.head_bias
