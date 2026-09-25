"""The context network: the committee network's tabular branch (prc2026_nn, unchanged encoders) plus a context branch.

Context branch: every neighbour token is described relative to the focal departure (times minus the focal takeoff t0 and minus
the focal NM off-block clock, the token's own durations, same-runway / same-stand / same-stand-group flags, arrival flag, ADS-B
state, wake category), encoded by a small ReLU MLP, then pooled two ways: multi-head attention with a query computed from the
focal row's encoded inputs, and a masked mean. Both pooled vectors (layer-normalised) and the scaled token count join the focal
inputs in front of the unchanged MLP backbone. Padding tokens (index -1) are masked; a row without any token gets zeros.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from prc2026_nn.model import NumericEncoder, embedding_dim

# Relative features computed per (focal, token); see token_features. Scales: 1800 s for times, 600 s for stopped time.
TOKEN_FEATURES = ["is_arr", "rel_mvt", "rel_mvt_fa", "rel_aobt", "rel_aobt_fa", "rel_eobt", "rel_sched", "rel_inblock", "taxi_in", "proxy",
                  "rel_motion", "rel_motion_fa", "adsb_taxi", "first_seen", "adsb_matched", "adsb_gate", "adsb_stopped", "same_runway",
                  "same_stand", "same_group", "sched_delay"]
ADSB_TOKEN_FEATURES = ["rel_motion", "rel_motion_fa", "adsb_taxi", "first_seen", "adsb_matched", "adsb_gate", "adsb_stopped"]
N_WAKE = 5
CLIP = 10.0


def token_features(times: torch.Tensor, values: torch.Tensor, codes: torch.Tensor, self_rows: torch.Tensor, nb_rows: torch.Tensor,
                   use_adsb: bool = True, departures_only: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather the neighbours of a batch and describe them relative to the focal departure.

    times (M, 7) float64 [t_mvt, t_aobt, t_eobt, t_sched, t_inblock, t_motion, t_seg]; values (M, 5) float32 [is_arr, taxi_in,
    adsb_matched, adsb_gate, adsb_stopped]; codes (M, 4) int32 [runway, stand, group, wake]; self_rows (B,); nb_rows (B, K), -1 padded.
    Returns (B, K, F) float32 values with NaN set to 0, (B, K, F) NaN flags, (B, K) wake codes, (B, K) validity."""
    valid = nb_rows >= 0
    rows = nb_rows.clamp(min=0)
    t = times[rows]                                 # (B, K, 7)
    v = values[rows]
    c = codes[rows]
    focal_t = times[self_rows]                      # (B, 7)
    focal_c = codes[self_rows]
    t0 = focal_t[:, 0:1].unsqueeze(-1)              # (B, 1, 1)
    fa = focal_t[:, 1:2].unsqueeze(-1)
    mvt, aobt, eobt, sched, inblock, motion, seg = (t[..., j:j + 1] for j in range(7))
    s = 1800.0
    parts = {
        "is_arr": v[..., 0:1].double(),
        "rel_mvt": (mvt - t0) / s, "rel_mvt_fa": (mvt - fa) / s,
        "rel_aobt": (aobt - t0) / s, "rel_aobt_fa": (aobt - fa) / s,
        "rel_eobt": (eobt - t0) / s, "rel_sched": (sched - t0) / 3600.0,
        "rel_inblock": (inblock - t0) / s, "taxi_in": v[..., 1:2].double() / s,
        "proxy": (mvt - aobt) / s,
        "rel_motion": (motion - t0) / s, "rel_motion_fa": (motion - fa) / s,
        "adsb_taxi": (mvt - motion) / s, "first_seen": (mvt - seg) / s,
        "adsb_matched": v[..., 2:3].double(), "adsb_gate": v[..., 3:4].double(), "adsb_stopped": v[..., 4:5].double() / 600.0,
        "same_runway": ((c[..., 0] == focal_c[:, None, 0]) & (c[..., 0] >= 0)).unsqueeze(-1).double(),
        "same_stand": ((c[..., 1] == focal_c[:, None, 1]) & (c[..., 1] >= 0)).unsqueeze(-1).double(),
        "same_group": ((c[..., 2] == focal_c[:, None, 2]) & (c[..., 2] >= 0)).unsqueeze(-1).double(),
        "sched_delay": (mvt - sched) / 3600.0,
    }
    if not use_adsb:
        for name in ADSB_TOKEN_FEATURES:
            parts[name] = torch.full_like(parts[name], float("nan"))
    x = torch.cat([parts[name] for name in TOKEN_FEATURES], dim=-1).float()
    nan = torch.isnan(x)
    x = torch.nan_to_num(x, nan=0.0).clamp(-CLIP, CLIP)
    if departures_only:
        valid = valid & (v[..., 0] < 0.5)
    return x, nan.float(), c[..., 3].long().clamp(0, N_WAKE - 1), valid


class ContextNet(nn.Module):
    def __init__(self, edges: np.ndarray, n_bins: np.ndarray, cardinalities: list[int], *, d_num: int = 8, cat_cap: int = 16, width: int = 512,
                 depth: int = 3, dropout: float = 0.1, tok_dim: int = 96, heads: int = 4, max_tokens: int = 128, wake_dim: int = 4):
        super().__init__()
        self.numeric = NumericEncoder(edges, n_bins, d_num)
        self.embeddings = nn.ModuleList([nn.Embedding(c, embedding_dim(c, cat_cap)) for c in cardinalities])
        d_in = self.numeric.out_features + edges.shape[0] + sum(e.embedding_dim for e in self.embeddings)
        f = len(TOKEN_FEATURES)
        self.wake = nn.Embedding(N_WAKE, wake_dim)
        self.token = nn.Sequential(nn.Linear(2 * f + wake_dim, tok_dim), nn.ReLU(), nn.Linear(tok_dim, tok_dim), nn.ReLU())
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
        self.d_in = d_in

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, tok: torch.Tensor, tok_nan: torch.Tensor, tok_wake: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """(N, 1) predictions in the scaled target unit (one member, the committee network's output shape)."""
        h, nan = self.numeric(x_num)
        z = torch.cat([h, nan.float()] + [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)], dim=1)
        e = self.token(torch.cat([tok, tok_nan, self.wake(tok_wake)], dim=-1))            # (B, K, D)
        b, k, d = e.shape
        mask = valid.float()                                                               # (B, K)
        q = self.query(z).view(b, self.heads, self.dk)
        keys = self.key(e).view(b, k, self.heads, self.dk)
        vals = self.value(e).view(b, k, self.heads, self.dk)
        scores = torch.einsum("bhd,bkhd->bhk", q, keys) / math.sqrt(self.dk)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=-1) * mask.unsqueeze(1)                        # all-padding rows -> zero weights
        attn = torch.einsum("bhk,bkhd->bhd", weights, vals).reshape(b, d)
        count = mask.sum(dim=1, keepdim=True)
        mean = (e * mask.unsqueeze(-1)).sum(dim=1) / count.clamp(min=1.0)
        pooled = torch.cat([self.norm_attn(attn), self.norm_mean(mean), count / self.max_tokens], dim=1)
        return self.head(self.backbone(torch.cat([z, pooled], dim=1)))
