"""Committee mix with a network share and the networks-far-below gate (configs/thinned_regime_round_v1.json, candidate n85_nd1200).

mix = (1 - a) committee + a nets where both are finite, the committee otherwise (tools/run_adsb_nn_dev.mixed); with a gate, rows whose
committee sits more than `threshold_s` above the networks take `share_inside` instead of a. No label is read here.
"""
from __future__ import annotations

import numpy as np

NETWORK = {"ens_ctx_e24": 0.25, "ens_ctx_e24b": 0.25, "ens_ctx_e32": 0.5}  # v30's network table (configs/adsb_ctx_v5.json c_e24x32_e32h)
CONTROL_SHARE = 0.75


def network_table(detail) -> np.ndarray:
    return np.sum([w * detail[name].to_numpy(dtype=float) for name, w in NETWORK.items()], axis=0)


def mix(committee: np.ndarray, nets: np.ndarray, share: float, gate: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mix, gate mask)."""
    c, n = np.asarray(committee, dtype=float), np.asarray(nets, dtype=float)
    finite = np.isfinite(c) & np.isfinite(n)
    a = np.full(len(c), float(share))
    fire = np.zeros(len(c), dtype=bool)
    if gate is not None:
        fire = finite & (c - n > float(gate["threshold_s"]))
        a = np.where(fire, float(gate["share_inside"]), a)
    return np.where(finite, (1 - a) * c + a * n, c), fire


def candidate_spec(declaration: dict, name: str) -> tuple[float, dict | None]:
    spec = declaration["candidates"][name]
    if spec.get("scope", "all") != "all":
        raise ValueError("Only whole-scope candidates are wired for the ranking stack.")
    return float(spec.get("network_share", declaration["control"]["network_share"])), spec.get("gate")


__all__ = ["NETWORK", "CONTROL_SHARE", "network_table", "mix", "candidate_spec"]
