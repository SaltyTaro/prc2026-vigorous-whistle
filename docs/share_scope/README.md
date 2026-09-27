# Network share by ADS-B scope (27 September 2026) — [`screen.json`](screen.json): nothing passes, closed

`tools/v30_share_scope.py` rebuilds v30 per row on the four recipe months at network shares {0.5, 0.6, 0.75, 0.85, 1.0} (the 0.75
rebuild reproduces the recorded combined candidate to 1e-9 on every month) and scores every (share on matched, share on unmatched)
non-LIRF pair against v30 (0.75 everywhere). Single shares: 0.85 helps July/December but not January/November; 1.0 helps only December.

| pair (matched / unmatched) | Jan | Jul | Nov | Dec | mean | intervals < 0 |
|---|---:|---:|---:|---:|---:|---:|
| 0.75 / 1.0 | +0.037 | −0.072 | −0.014 | −0.242 | −0.073 | 1/4 |
| 0.85 / 1.0 | +0.042 | −0.051 | +0.040 | −0.290 | −0.065 | 1/4 |
| 0.75 / 0.85 | +0.001 | −0.054 | −0.031 | −0.132 | −0.054 | 2/4 |
| 0.6 / 0.85 | +0.009 | −0.066 | −0.062 | −0.007 | −0.032 | 2/4 |

No pair has every month at or below v30 with intervals below zero (the unchanged rule); the gains are December-driven. Closed.
