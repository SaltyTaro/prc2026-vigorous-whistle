"""One more member in the committee's network table (tools/run_network_member.py, tools/export_network_member.py).

table' = (1 - w) table + w member where both are finite, the table elsewhere, so rows the member does not cover keep the shipped table.
The mix on top of table' is v46's (src/prc2026_final/share_gate.py). No label is read here.
"""
from __future__ import annotations

import numpy as np

V46_SHARE = 0.85
V46_GATE = {"threshold_s": 1200, "share_inside": 0.5}  # configs/thinned_regime_round_v1.json n85_nd1200


def seed_mean(tables: list) -> np.ndarray:
    """Row mean over seeds; a row missing in any seed is missing (no partial ensembles)."""
    if not tables:
        raise ValueError("No seed tables.")
    stacked = np.vstack([np.asarray(t, dtype=float) for t in tables])
    return np.where(np.isfinite(stacked).all(axis=0), stacked.mean(axis=0), np.nan)


def table_with_member(table: np.ndarray, member: np.ndarray, weight: float) -> tuple[np.ndarray, np.ndarray]:
    """(table', rows where the member enters)."""
    if not 0.0 <= float(weight) <= 1.0:
        raise ValueError("The member weight lies in [0, 1].")
    t, m = np.asarray(table, dtype=float), np.asarray(member, dtype=float)
    inside = np.isfinite(t) & np.isfinite(m)
    return np.where(inside, (1 - float(weight)) * t + float(weight) * m, t), inside


def candidate_weight(declaration: dict, name: str) -> float:
    spec = declaration["candidates"][name]
    if set(spec) - {"member_weight", "note"}:
        raise ValueError(f"{name}: only member_weight is wired ({sorted(spec)}).")
    return float(spec["member_weight"])


__all__ = ["V46_SHARE", "V46_GATE", "seed_mean", "table_with_member", "candidate_weight"]
