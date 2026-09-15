"""Flight-group cross-fitted, partially pooled stand/runway target baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_phase4.features import prediction_offset

BASELINE_NAMES = ["taxi_base_q20_s", "taxi_base_mean_s", "taxi_base_iqr_s",
                  "taxi_base_support", "taxi_base_level", "taxi_airport_scale_s"]
KEY_LEVELS = [["airport"], ["airport", "nm", "aircraft_flag"],
              ["airport", "nm", "aircraft_flag", "runway"],
              ["airport", "nm", "aircraft_flag", "stand"],
              ["airport", "nm", "aircraft_flag", "stand", "runway"]]
STATS = ["count", "mean", "q20", "q25", "q75"]


def aligned_truth(features: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    if len(features) != len(truth) or features[ID].duplicated().any() or truth[ID].duplicated().any():
        raise ValueError("Target rows must align one-to-one with features.")
    joined = features[[ID]].merge(truth, on=ID, how="left", validate="one_to_one")
    if not np.isfinite(joined[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Missing/nonfinite aligned labels.")
    return joined


def _keys(features: pd.DataFrame) -> pd.DataFrame:
    out = features[["airport", "stand", "runway"]].astype("string").fillna("__MISSING__").reset_index(drop=True)
    out["nm"] = np.where(features["nm_missing"].eq(1), "1", "0")
    out["aircraft_flag"] = np.where(features["aircraft_missing"].eq(1), "1", "0")
    return out


def fit_map(features: pd.DataFrame, residual: np.ndarray, spec: dict) -> dict:
    residual = np.asarray(residual, dtype=float)
    if len(features) != len(residual) or not len(features) or not np.isfinite(residual).all():
        raise ValueError("Baseline fitting requires every finite fitting target.")
    if len(spec["shrinkage"]) != len(KEY_LEVELS) or any(n <= 0 for n in spec["shrinkage"]):
        raise ValueError("Positive shrinkage is required at every baseline level.")
    data = _keys(features)
    data["value"] = residual
    quantiles = np.quantile(residual, [.2, .25, .75])
    global_stats = [len(residual), float(np.mean(residual)), *map(float, quantiles)]
    levels = []
    for keys, strength in zip(KEY_LEVELS, spec["shrinkage"]):
        group = data.groupby(keys, sort=True, observed=True)["value"]
        stats = group.agg(["count", "mean"])
        q = group.quantile([.2, .25, .75]).unstack()
        stats[["q20", "q25", "q75"]] = q[[.2, .25, .75]]
        table = stats.reset_index()
        records = [[*(str(row[k]) for k in keys), int(row["count"]),
                    *(float(row[k]) for k in STATS[1:])] for row in table.to_dict("records")]
        levels.append({"keys": keys, "shrinkage": float(strength), "records": records})
    return {"policy": "flight-crossfit-pooled-q20-v1", "spec": spec, "global": global_stats,
            "levels": levels, "fit_rows": len(features), "fit_ids_sha256": identity_hash(features[ID])}


def apply_map(features: pd.DataFrame, model: dict) -> pd.DataFrame:
    if model["policy"] != "flight-crossfit-pooled-q20-v1":
        raise ValueError("Unknown baseline policy.")
    keys = _keys(features)
    stats = np.tile(np.asarray(model["global"][1:], dtype=float), (len(features), 1))
    support = np.zeros(len(features), dtype=float)
    level = np.zeros(len(features), dtype=float)
    airport_scale = np.full(len(features), max(model["spec"]["minimum_scale_seconds"], stats[0, 3]-stats[0, 2])) if len(features) else np.empty(0)
    for index, node in enumerate(model["levels"], start=1):
        table = pd.DataFrame(node["records"], columns=[*node["keys"], *STATS])
        aligned = keys[node["keys"]].merge(table, on=node["keys"], how="left", validate="many_to_one")
        values = aligned[STATS].to_numpy(dtype=float)
        count = values[:, 0]
        valid = np.isfinite(count) & (count > 0)
        weight = count[valid]/(count[valid]+node["shrinkage"])
        stats[valid] = (1-weight[:, None])*stats[valid]+weight[:, None]*values[valid, 1:]
        support[valid], level[valid] = count[valid], index
        if index == 1:
            airport_scale = np.maximum(model["spec"]["minimum_scale_seconds"], stats[:, 3]-stats[:, 2])
    out = pd.DataFrame({"taxi_base_q20_s": stats[:, 1], "taxi_base_mean_s": stats[:, 0],
                        "taxi_base_iqr_s": np.maximum(0, stats[:, 3]-stats[:, 2]),
                        "taxi_base_support": support, "taxi_base_level": level,
                        "taxi_airport_scale_s": airport_scale})
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Baseline fallback must produce finite features.")
    return out.astype("float32")


def group_folds(features: pd.DataFrame, truth: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    if folds < 2 or "FLIGHT_ID_mvt" not in truth:
        raise ValueError("Cross-fitting needs at least two folds and flight grouping metadata.")
    meta = aligned_truth(features, truth)
    groups = meta["FLIGHT_ID_mvt"].astype("string")
    groups = ("flight:"+groups).where(groups.notna(), "movement:"+meta[ID].astype("int64").astype("string"))
    hashed = pd.util.hash_pandas_object(groups, index=False).to_numpy(dtype="uint64")
    result = ((hashed ^ np.uint64(seed)) % np.uint64(folds)).astype("int8")
    if len(np.unique(result)) != folds:
        raise ValueError("Every inner cross-fitting fold must have fitting and held-out rows.")
    return result


def cross_fit(features: pd.DataFrame, truth: pd.DataFrame, spec: dict):
    meta = aligned_truth(features, truth)
    residual = meta[TARGET].to_numpy(dtype=float)-prediction_offset(features, "lirf_missing_schedule")
    folds = group_folds(features, truth, spec["inner_folds"], spec["seed"])
    out = pd.DataFrame(np.nan, index=np.arange(len(features)), columns=BASELINE_NAMES, dtype="float32")
    receipts = []
    for fold in range(spec["inner_folds"]):
        fit, hold = folds != fold, folds == fold
        model = fit_map(features.loc[fit], residual[fit], spec)
        out.loc[hold, BASELINE_NAMES] = apply_map(features.loc[hold], model).to_numpy()
        receipts.append({"inner_fold": fold, "fit_rows": int(fit.sum()), "heldout_rows": int(hold.sum()),
                         "fit_ids_sha256": identity_hash(features.loc[fit, ID]),
                         "heldout_ids_sha256": identity_hash(features.loc[hold, ID])})
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("Cross-fitting did not cover every fitting row.")
    full = fit_map(features, residual, spec)
    assignments = pd.DataFrame({ID: features[ID].to_numpy(), "inner_fold": folds})
    return out, full, {"folds": receipts, "grouping": "flight ID; movement ID only for null flight IDs",
                       "all_fitting_labels_retained": True}, assignments


def center_scale(baseline: pd.DataFrame | None, policy: str, rows: int):
    if policy == "none":
        return np.zeros(rows), np.ones(rows), np.ones(rows)
    if baseline is None or len(baseline) != rows or policy not in {"baseline", "baseline_scaled"}:
        raise ValueError("Invalid residual target transform.")
    center = baseline["taxi_base_q20_s"].to_numpy(dtype=float)
    scale = baseline["taxi_airport_scale_s"].to_numpy(dtype=float) if policy == "baseline_scaled" else np.ones(rows)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Target scales must be positive and finite.")
    # A fixed common divisor does not alter relative squared-seconds loss.
    weights = (scale/600.0)**2 if policy == "baseline_scaled" else np.ones(rows)
    return center, scale, weights
