"""Frozen date/flight-group splits and complete-population evaluation."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd

from .features import ID, TIME, TARGET


def identity_hash(ids) -> str:
    """Order-independent, platform-stable identity of a set of integral IDs."""
    values = np.asarray(ids, dtype=np.float64)
    if not np.isfinite(values).all() or (values % 1 != 0).any():
        raise ValueError("Invalid movement IDs.")
    return hashlib.sha256(np.sort(values.astype("<i8")).tobytes()).hexdigest()


def validate_protocol(protocol: dict) -> None:
    if protocol["assignment_clock"] != TIME or protocol["scored_phase"] != "DEP":
        raise ValueError("Unsupported partition clock or phase.")
    if (protocol["group_key"] != "FLIGHT_ID_mvt"
            or protocol["null_group_policy"] != "treat_each_movement_as_an_independent_group"
            or protocol["group_priority"] != ["score", "early_stop", "fit"]
            or protocol["target_policy"] != "retain_all_finite_labels_in_every_evaluation"):
        raise ValueError("Unsupported group or target policy.")
    reserved = set(protocol["reserved_audit_months"])
    if not reserved:
        raise ValueError("An audit period must be reserved.")
    for name, spec in protocol["folds"].items():
        roles = [set(spec[f"{role}_months"]) for role in ["fit", "early_stop", "score"]]
        if any(not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month)
               for month in set.union(reserved, *roles)):
            raise ValueError("Months must use YYYY-MM.")
        if any(not role for role in roles):
            raise ValueError(f"Empty month definition in {name}.")
        if any(roles[a] & roles[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError(f"Overlapping partition months in {name}.")
        if not spec["sealed"] and any(role & reserved for role in roles):
            raise ValueError("A development fold uses the reserved audit period.")
        if spec["sealed"] and roles[2] != reserved:
            raise ValueError("Final-audit score months must be the reserved period.")
        if name in {"forward", "final_audit"} and not (max(roles[0]) < min(roles[1]) <= max(roles[1]) < min(roles[2])):
            raise ValueError(f"Non-chronological {name} fold.")


def assign_fold(catalog: pd.DataFrame, protocol: dict, fold: str):
    validate_protocol(protocol)
    spec = protocol["folds"][fold]
    if catalog[ID].isna().any() or catalog[ID].duplicated().any():
        raise ValueError("Duplicate or null catalog movement IDs.")
    result = catalog[[ID, "FLIGHT_ID_mvt", TIME, "airport", "source_file"]].copy()
    result["month"] = result[TIME].dt.strftime("%Y-%m")
    result["partition"] = "unused"
    for role in ["fit", "early_stop", "score"]:
        result.loc[result["month"].isin(spec[f"{role}_months"]), "partition"] = role
    if not spec["sealed"]:
        reserved_mask = result["month"].isin(protocol["reserved_audit_months"])
        result.loc[reserved_mask, "partition"] = "reserved_audit"
        reserved_groups = set(result.loc[reserved_mask, "FLIGHT_ID_mvt"].dropna())
    else:
        reserved_groups = set()
    score_groups = set(result.loc[result["partition"].eq("score"), "FLIGHT_ID_mvt"].dropna())
    # Protect the complete evaluation population; purge only fitting/stopping rows.
    early = result["partition"].eq("early_stop")
    purge = early & result["FLIGHT_ID_mvt"].isin(score_groups | reserved_groups)
    result.loc[purge, "partition"] = "purged_early_stop"
    stop_groups = set(result.loc[result["partition"].eq("early_stop"), "FLIGHT_ID_mvt"].dropna())
    fit = result["partition"].eq("fit")
    purge = fit & result["FLIGHT_ID_mvt"].isin(score_groups | stop_groups | reserved_groups)
    result.loc[purge, "partition"] = "purged_fit"
    report = {"name": fold, "definition": spec, "counts": {}, "identity_sha256": {}, "group_overlap_counts": {}}
    for role, part in result.groupby("partition"):
        report["counts"][role] = len(part)
        report["identity_sha256"][role] = identity_hash(part[ID])
    for a, b in [("fit", "early_stop"), ("fit", "score"), ("early_stop", "score")]:
        ga = set(result.loc[result["partition"].eq(a), "FLIGHT_ID_mvt"].dropna())
        gb = set(result.loc[result["partition"].eq(b), "FLIGHT_ID_mvt"].dropna())
        overlap = len(ga & gb)
        report["group_overlap_counts"][f"{a}/{b}"] = overlap
        if overlap:
            raise ValueError(f"Flight-group overlap after purging: {fold}/{a}/{b}")
    expected = catalog.loc[catalog[TIME].dt.strftime("%Y-%m").isin(spec["score_months"]), ID]
    actual = result.loc[result["partition"].eq("score"), ID]
    if len(expected) != len(actual) or identity_hash(expected) != identity_hash(actual):
        raise ValueError("Evaluation population changed during group purging.")
    report["score_groups_also_in_reserved_audit"] = len(score_groups & reserved_groups)
    return result, report


def require_development_fold(protocol: dict, fold: str) -> None:
    if fold not in protocol["folds"]:
        raise ValueError(f"Unknown fold: {fold}")
    if protocol["folds"][fold]["sealed"]:
        raise ValueError("The final audit is sealed. Phase 1 cannot score it.")


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = pred.astype(np.float64) - y.astype(np.float64)
    sse = float(np.dot(error, error))
    return {"rows": len(y), "sse": sse, "rmse_seconds": float(np.sqrt(sse / len(y))),
            "mae_seconds": float(np.abs(error).mean()), "max_absolute_error_seconds": float(np.abs(error).max())}


def evaluate(truth: pd.DataFrame, predictions: pd.DataFrame, seasonal_weights=None) -> dict:
    for name, frame in [("truth", truth), ("predictions", predictions)]:
        if frame.empty or frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"Empty data or duplicate/null IDs in {name}.")
    if len(truth) != len(predictions) or set(truth[ID]) != set(predictions[ID]):
        raise ValueError("Prediction IDs must match the complete evaluation set exactly.")
    if not np.isfinite(truth[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite evaluation label; cannot silently drop rows.")
    if not np.isfinite(predictions["prediction"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite prediction.")
    aligned = truth.merge(predictions[[ID, "prediction"]], on=ID, how="left", validate="one_to_one")
    result = {"overall": _metrics(aligned[TARGET].to_numpy(), aligned["prediction"].to_numpy()),
              "evaluation_ids_sha256": identity_hash(aligned[ID])}
    aligned["month"] = aligned[TIME].dt.strftime("%Y-%m")
    aligned["date"] = aligned[TIME].dt.strftime("%Y-%m-%d")
    y = aligned[TARGET]
    aligned["target_band"] = np.select([y < 0, y.eq(0), y <= 3600, y <= 10800],
                                        ["negative", "zero", "0_to_1h", "1h_to_3h"], default="over_3h")
    groups = ["airport", "month", "date", "target_band"]
    if "nm_missing" in aligned:
        groups.append("nm_missing")
    for name in groups:
        result[f"by_{name}"] = {str(key): _metrics(g[TARGET].to_numpy(), g["prediction"].to_numpy())
                                 for key, g in aligned.groupby(name, dropna=False, observed=True)}
    if seasonal_weights is not None:
        observed_months = set(aligned[TIME].dt.month)
        weights = {int(k): float(v) for k, v in seasonal_weights.items()}
        if (observed_months != set(weights) or not np.isclose(sum(weights.values()), 1)
                or any(not np.isfinite(v) or v < 0 for v in weights.values())):
            raise ValueError("Seasonal weights must cover exactly the scored months and sum to one.")
        weighted_mse = 0.0
        for month, weight in weights.items():
            part = aligned.loc[aligned[TIME].dt.month.eq(month)]
            m = _metrics(part[TARGET].to_numpy(), part["prediction"].to_numpy())
            weighted_mse += weight * m["sse"] / m["rows"]
        result["ranking_month_mix_rmse_seconds"] = float(np.sqrt(weighted_mse))
        result["ranking_month_weights"] = weights
    # Ensure saved reports cannot silently contain JSON NaN/Infinity.
    json.dumps(result, allow_nan=False)
    return result
