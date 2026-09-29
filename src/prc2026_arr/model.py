"""The plain network with a second linear head for the arrival taxi-in time.

Base's parameters are created first, in base's order, so a seed gives base's initial network; the arrival head is drawn afterwards.
`forward` is base's (the departure head). The task of a row is visible to the network through the missing-value flags of the
departure-only columns; there is no task input.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from prc2026_nn.model import TabularNet


class TwoTaskNet(TabularNet):
    def __init__(self, edges, n_bins, cardinalities, **architecture):
        super().__init__(edges, n_bins, cardinalities, **architecture)
        width = self.head_weight.shape[1]
        self.arr_weight = nn.Parameter(torch.empty(self.k, width))
        self.arr_bias = nn.Parameter(torch.zeros(self.k))
        nn.init.uniform_(self.arr_weight, -1.0 / math.sqrt(width), 1.0 / math.sqrt(width))

    def hidden(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """(N, k, width): base's forward up to the head."""
        h, nan = self.numeric(x_num)
        parts = [h, nan.float()] + [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
        z = torch.cat(parts, dim=1)
        n = z.shape[0]
        if self.k > 1:
            z = (z.unsqueeze(1) * self.adapter).reshape(n * self.k, self.d_in)
        return self.backbone(z).reshape(n, self.k, -1)

    def heads(self, x_num: torch.Tensor, x_cat: torch.Tensor, split: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One pass over a batch whose first `split` rows are departures and whose other rows are arrivals: each part through its own head."""
        z = self.hidden(x_num, x_cat)
        departures = torch.einsum("nkw,kw->nk", z[:split], self.head_weight) + self.head_bias
        arrivals = torch.einsum("nkw,kw->nk", z[split:], self.arr_weight) + self.arr_bias
        return departures, arrivals

    def arrival(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """(N, k) arrival-head predictions in the scaled arrival unit."""
        return torch.einsum("nkw,kw->nk", self.hidden(x_num, x_cat), self.arr_weight) + self.arr_bias
