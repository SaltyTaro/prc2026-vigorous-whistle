"""The two network members v51 and v52 ship, on any file's rows (stage `members`; layer sets v51m and v52m in stack.py).

- combo_all (configs/adsb_ctx3_combo_dev_v1.json, w40 selected by its rule): the dl32 seed mean on rows with adsb_matched = 1 and the
  m25dl32 seed mean on the other rows (16 seeds each, twelve-month refits).
- mask_unmatched (configs/adsb_ctx3_mask_dev_v1.json, w40 by the user's override): the m25_32 seed mean on rows with adsb_matched = 0;
  rows with a match carry no member value.
They enter the network table in that order at 0.40 each on non-LIRF rows where both are finite (src/prc2026_final/network_member.py),
as tools/export_network_member.py built v51's and v52's member files. The seed mean of the routed per-seed tables (how the exports
read them) equals the routing of the per-variant seed means, row for row. v51 and v52 then add the published daily totals
(tools/export_daily_totals.py --base), which is not part of this runner: v51m and v52m are the open-data model parts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json
from prc2026.features import ID

from . import network_member
from .common import ROOT

COMBO, MASK = "configs/adsb_ctx3_combo_dev_v1.json", "configs/adsb_ctx3_mask_dev_v1.json"
ENSEMBLES = {"combo_dl32": (COMBO, "dl32"), "combo_m25dl32": (COMBO, "m25dl32"), "mask_m25_32": (MASK, "m25_32")}
WEIGHTS = {"combo_all": 0.40, "mask_unmatched": 0.40}


def learners() -> dict[str, list[str]]:
    """ensemble -> the declared learners' directories (each holds refit12/)."""
    out = {}
    for name, (path, variant) in ENSEMBLES.items():
        fits = read_json(ROOT / path)["fits"]
        if variant not in fits["variants"]:
            raise ValueError(f"{path} does not declare {variant}.")
        base = fits["outputs"].split("/nets/")[0]
        out[name] = [f"{base}/nets/{variant}_s{seed}" for seed in fits["seeds"]]
    return out


def members(ids: np.ndarray, ensembles: pd.DataFrame, matched: np.ndarray) -> pd.DataFrame:
    """The two member columns from the three seed means (`ensembles` in `ids` order) and the file's match flag."""
    if not np.array_equal(ensembles[ID].to_numpy(), ids):
        raise ValueError("The member ensembles are not in the file's departure order.")
    matched = np.asarray(matched, dtype=bool)
    combo = np.where(matched, ensembles["combo_dl32"].to_numpy(dtype=float), ensembles["combo_m25dl32"].to_numpy(dtype=float))
    mask = np.where(matched, np.nan, ensembles["mask_m25_32"].to_numpy(dtype=float))
    return pd.DataFrame({ID: ids, "adsb_matched": matched, "combo_all": combo, "mask_unmatched": mask})


def tables(nets: np.ndarray, frame: pd.DataFrame, lirf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(v51's table, rows the combination enters, v52's table, rows the masked member enters); LIRF rows keep the shipped table."""
    lirf = np.asarray(lirf, dtype=bool)
    t51, in51 = network_member.table_with_member(nets, np.where(lirf, np.nan, frame["combo_all"].to_numpy(dtype=float)), WEIGHTS["combo_all"])
    t52, in52 = network_member.table_with_member(t51, np.where(lirf, np.nan, frame["mask_unmatched"].to_numpy(dtype=float)), WEIGHTS["mask_unmatched"])
    return t51, in51, t52, in52


__all__ = ["COMBO", "MASK", "ENSEMBLES", "WEIGHTS", "learners", "members", "tables"]
