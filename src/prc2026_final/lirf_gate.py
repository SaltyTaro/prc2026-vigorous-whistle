"""LIRF disagreement gate (configs/lirf_gate_v1.json): lean to the LIRF tree where it sits far above the LIRF network.

The v34 LIRF non-population blends are v9 / network / tree weighted sums. On rows of a scope whose tree - network exceeds T, the gate
replaces the blend by a declared leaning blend; every other row keeps the v34 blend. No label is read here.
"""
from __future__ import annotations

import numpy as np

FLOOR = -12.0
V34 = {"unmatched": {"v9": 0.0, "net": 0.35 / 0.60, "tree": 0.25 / 0.60}, "matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}}


def blend(v9: np.ndarray, net: np.ndarray, tree: np.ndarray, weights: dict) -> np.ndarray:
    w = weights
    if abs(w["v9"] + w["net"] + w["tree"] - 1.0) > 1e-9 or min(w.values()) < 0:
        raise ValueError("Blend weights must be non-negative and sum to one.")
    return np.maximum(w["v9"] * v9 + w["net"] * net + w["tree"] * tree, FLOOR)


def gated(v9: np.ndarray, net: np.ndarray, tree: np.ndarray, scope: str, spec: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """(values, gate mask) for rows of one scope: v34's blend, replaced by the leaning blend where tree - net > T."""
    v9, net, tree = (np.asarray(a, dtype=float) for a in (v9, net, tree))
    base = blend(v9, net, tree, V34[scope])
    if spec is None:
        return base, np.zeros(len(base), dtype=bool)
    fire = (tree - net) > float(spec["T"])
    values = base.copy()
    values[fire] = blend(v9[fire], net[fire], tree[fire], spec["lean"])
    return values, fire


def paired_day_interval(delta_by_day: np.ndarray, total: float, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(delta_by_day)
    sums = np.array([delta_by_day[rng.integers(0, n, n)].sum() for _ in range(draws)])
    return float(np.quantile(sums, 0.025) / total), float(np.quantile(sums, 0.975) / total)


def evaluate(v9, net, tree, labels, days, scope: str, spec: dict | None, cap_s: float, draws: int, seed: int) -> dict:
    """SSE change of the gate against v34's blend on one scope: primary (labels below cap_s) and unfiltered, with the paired-day interval."""
    labels, days = np.asarray(labels, dtype=float), np.asarray(days)
    base, _ = gated(v9, net, tree, scope, None)
    values, fire = gated(v9, net, tree, scope, spec)
    out = {"rows": int(len(labels)), "gated_rows": int(fire.sum())}
    for name, keep in (("primary", labels < cap_s), ("unfiltered", np.ones(len(labels), dtype=bool))):
        e0, e1 = (base - labels) ** 2, (values - labels) ** 2
        total = float(e0[keep].sum())
        by_day = np.array([float(e1[keep & (days == d)].sum() - e0[keep & (days == d)].sum()) for d in np.unique(days)])
        lo, hi = paired_day_interval(by_day, total, draws, seed) if total > 0 and len(by_day) else (0.0, 0.0)
        out[name] = {"sse_control": total, "sse_change": float(e1[keep].sum() - e0[keep].sum()), "relative_change": float((e1[keep].sum() - e0[keep].sum()) / total) if total > 0 else 0.0,
                     "interval_95": [lo, hi], "days_better": int((by_day < 0).sum()), "days": int(len(by_day)), "rows_kept": int(keep.sum()),
                     "gated_rows_rmse_control": float(np.sqrt(e0[keep & fire].mean())) if (keep & fire).any() else None,
                     "gated_rows_rmse_gate": float(np.sqrt(e1[keep & fire].mean())) if (keep & fire).any() else None}
    return out


__all__ = ["FLOOR", "V34", "blend", "gated", "evaluate", "paired_day_interval"]
