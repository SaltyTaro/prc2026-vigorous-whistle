"""Neighbour selection with a configurable window: the first round's rule (prc2026_ctx.tokens.neighbours) with its constants
as parameters. Tokens themselves come from prc2026_ctx.tokens.build_tokens / encode_codes, unchanged (same label contract).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026_ctx.tokens import ID


def neighbours_window(tokens: pd.DataFrame, k: int, before_s: float, after_s: float, centre_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(departure ids, their token rows, (n_dep, k) int32 neighbour rows padded with -1): movements at the same airport with time in
    [t0 - before_s, t0 + after_s], the focal row excluded, at most k, a contiguous time run centred on t0 + centre_s."""
    rows = np.flatnonzero(tokens["is_dep"].to_numpy())
    t = tokens["t_mvt"].to_numpy()
    airport = tokens["airport"].to_numpy()
    index = np.full((len(rows), k), -1, dtype=np.int32)
    for span in (np.flatnonzero(airport == a) for a in pd.unique(airport)):
        lo_a, hi_a = span[0], span[-1] + 1
        mine = rows[(rows >= lo_a) & (rows < hi_a)]
        if len(mine) == 0:
            continue
        ta = t[lo_a:hi_a]
        t0 = t[mine]
        lo = lo_a + np.searchsorted(ta, t0 - before_s, side="left")
        hi = lo_a + np.searchsorted(ta, t0 + after_s, side="right")
        centre = lo_a + np.searchsorted(ta, t0 + centre_s, side="left")
        start = np.clip(centre - (k + 1) // 2, lo, np.maximum(lo, hi - (k + 1)))
        cand = start[:, None] + np.arange(k + 1)[None, :]
        ok = (cand < hi[:, None]) & (cand != mine[:, None])
        order = np.argsort(~ok, axis=1, kind="stable")[:, :k]
        picked = np.take_along_axis(cand, order, axis=1)
        valid = np.take_along_axis(ok, order, axis=1)
        index[np.searchsorted(rows, mine)] = np.where(valid, picked, -1).astype(np.int32)
    return tokens[ID].to_numpy()[rows], rows.astype(np.int64), index
