"""Variance re-allocation of the airport-day's published discrepancy (a layer after the extreme-day layer; configs/scout_var_alloc_v1.json).

v48 moves every ordinary row of an airport-day by the same posterior day shift plus a first-order tilt sized by the row's cell. This
layer re-allocates the day's discrepancy between rows by a learned, label-free estimate of each row's error variance and chance of a
visible delay:

  I_d    the extreme-day layer's implied shift of the airport-day (in-figure rows, c0 offset), clipped to [-CLIP_S, CLIP_S]
  w_i    exp(variance model) x P(late): LightGBM of log(r^2 + 100) and of (a > label), r = label - the layer's input value
  wr_i   w_i / mean of w over the eligible rows of the airport-day
  c_i    clip(k x I_d x (wr_i - 1), -CAP_S, CAP_S), k the least-squares slope of r on I_d x (wr - 1) over fitting rows
Eligible rows: in the figure, moved by v48 (heavy rows and rows of unread days keep their value), not the LIRF population; the
caller adds guard rows. Before the cap the changes of an airport-day sum to zero: the day's level stays the earlier layers'.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

CAP_S, CLIP_S, R_MAX_S, FLOOR_S = 100.0, 600.0, 1800.0, -12.0
ROUNDS = 300
PARAMS = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 2000, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 20261002, "num_threads": 6}
RAW_COLUMNS = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt"]
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WAKE = ["H", "J", "L", "M"]
SEGMENTS = ["Business", "Cargo", "Lowcost", "Mainline", "Military", "Non-Scheduled", "Other", "Regional"]
FEATURES = ["airport", "matched", "a", "stack", "slack", "hour", "has_aobt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "wtc", "seg", "n_day"]


def _codes(values: pd.Series, vocabulary: list[str]) -> np.ndarray:
    return pd.Categorical(values, categories=vocabulary).codes.astype(np.int64)


def features(t: pd.DataFrame, raw: pd.DataFrame, day_keys: list[str]) -> pd.DataFrame:
    """Label-free inputs per row of t (needs MVT_ID_mvt, airport, matched, a, stack and day_keys); raw: departures indexed by MVT_ID_mvt."""
    r = raw.reindex(t["MVT_ID_mvt"].to_numpy())
    tm = pd.to_datetime(r["MVT_TIME_UTC_mvt"].to_numpy(), utc=True)
    f = pd.DataFrame(index=t.index)
    f["airport"] = _codes(t["airport"], AIRPORTS)
    f["matched"] = t["matched"].astype(int).to_numpy()
    f["a"], f["stack"] = t["a"].to_numpy(dtype=float), t["stack"].to_numpy(dtype=float)
    f["slack"] = f["a"] - f["stack"]
    f["hour"] = tm.hour + tm.minute / 60
    f["has_aobt"] = r["AOBT_3_flt"].notna().to_numpy().astype(int)
    for column, ref in (("AOBT_3_flt", "stack"), ("EOBT_1_flt", "a"), ("LOBT_flt", "stack")):
        f[column] = (tm - pd.to_datetime(r[column].to_numpy(), utc=True)).total_seconds().to_numpy() - f[ref].to_numpy()
    f["wtc"] = _codes(r["WK_TBL_CAT_flt"], WAKE)
    f["seg"] = _codes(r["MARKET_SEGMENT_flt"], SEGMENTS)
    f["n_day"] = t.groupby(day_keys)["a"].transform("size").to_numpy()
    return f[FEATURES]


def eligible(t: pd.DataFrame, guard=()) -> np.ndarray:
    return (t["moved"].astype(bool) & t["in_figure"].astype(bool) & ~t["cls"].eq("LIRF_pop") & ~t["MVT_ID_mvt"].isin(list(guard))).to_numpy()


def fit_models(f: pd.DataFrame, a: np.ndarray, label: np.ndarray, r: np.ndarray, rows: np.ndarray) -> tuple:
    """Variance and late models on the fitting rows (rows: boolean mask of eligible fitting rows)."""
    ok = rows & (np.abs(r) < R_MAX_S)
    variance = lgb.train(PARAMS, lgb.Dataset(f.loc[ok], np.log(r[ok] ** 2 + 100.0)), ROUNDS)
    late = lgb.train({**PARAMS, "objective": "binary"}, lgb.Dataset(f.loc[rows], (a[rows] > label[rows]).astype(int)), ROUNDS)
    return variance, late


def weights(models: tuple, f: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    variance, late = models
    w = np.full(len(f), np.nan)
    if rows.any():
        w[rows] = np.exp(variance.predict(f.loc[rows])) * late.predict(f.loc[rows])
    return w


def basis(t: pd.DataFrame, w: np.ndarray, I: np.ndarray, day_keys: list[str]) -> np.ndarray:
    """I_d x (wr - 1) per row; 0 where the row is not eligible (w NaN)."""
    s = pd.Series(w, index=t.index)
    wr = s / s.groupby([t[k] for k in day_keys]).transform("mean")
    return np.nan_to_num((np.clip(I, -CLIP_S, CLIP_S) * (wr - 1.0)).to_numpy(dtype=float), nan=0.0)


def slope(r: np.ndarray, base: np.ndarray, rows: np.ndarray) -> float:
    ok = rows & (np.abs(r) < R_MAX_S)
    den = float((base[ok] ** 2).sum())
    return float((r[ok] * base[ok]).sum() / den) if den > 0 else 0.0


def update(value: np.ndarray, k: float, base: np.ndarray, rows: np.ndarray) -> np.ndarray:
    out = np.asarray(value, dtype=float).copy()
    out[rows] = np.maximum(out[rows] + np.clip(k * base[rows], -CAP_S, CAP_S), FLOOR_S)
    return out


__all__ = ["CAP_S", "CLIP_S", "R_MAX_S", "FLOOR_S", "FEATURES", "features", "eligible", "fit_models", "weights", "basis", "slope", "update"]
