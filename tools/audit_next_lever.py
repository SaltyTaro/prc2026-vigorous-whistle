"""Audit unused flown-arrival clocks on the common fitting population only."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_nmclock.bundle import load_bundle
from prc2026_phase3.bundle import load_partition

RAW = [ID, PHASE, TIME, "ADEP_mvt", "ARVT_3_flt", "ARVT_1_flt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt"]
BASE = ["airport", "nmclock_available", "nmclock_proxy_s", "conditional_schedule_offset_s"]


def unused_clocks(raw):
    return pd.DataFrame({
        ID: raw[ID].to_numpy(),
        "flown_arrival_minus_takeoff_s": (raw["ARVT_3_flt"]-raw[TIME]).dt.total_seconds(),
        "flown_arrival_minus_plan_s": (raw["ARVT_3_flt"]-raw["ARVT_1_flt"]).dt.total_seconds(),
        "flown_duration_s": (raw["ARVT_3_flt"]-raw["AOBT_3_flt"]).dt.total_seconds(),
        "flown_lobt_duration_s": (raw["ARVT_3_flt"]-raw["LOBT_flt"]).dt.total_seconds(),
        "flown_minus_planned_duration_s": ((raw["ARVT_3_flt"]-raw["AOBT_3_flt"])-(raw["ARVT_1_flt"]-raw["EOBT_1_flt"])).dt.total_seconds(),
        "flown_minus_initial_duration_s": ((raw["ARVT_3_flt"]-raw["AOBT_3_flt"])-(raw["ARVT_1_flt"]-raw["IOBT_flt"])).dt.total_seconds(),
    })


def summary(frame):
    available = frame.nmclock_available.eq(1)
    out = {"rows": len(frame), "clock_present_rows": int(available.sum()), "flown_arrival_present_rows": int(frame.flown_arrival_minus_takeoff_s.notna().sum())}
    if "residual" not in frame:
        return out
    use = frame.loc[available].copy()
    r = use.residual.to_numpy(dtype=float)
    center = use.groupby("airport", observed=True).residual.transform("mean").to_numpy()
    r = r-center
    out["airport_centered_raw_proxy_residual_rmse_seconds"] = float(np.sqrt(np.mean(r*r)))
    out["correlations_with_airport_centered_proxy_residual"] = {}
    for name in unused_clocks_columns:
        x = use[name].to_numpy(dtype=float)
        valid = np.isfinite(x)
        if valid.sum() > 20 and np.std(x[valid]) > 0:
            out["correlations_with_airport_centered_proxy_residual"][name] = {
                "pearson": float(np.corrcoef(x[valid], r[valid])[0, 1]),
                "spearman": float(pd.Series(x[valid]).corr(pd.Series(r[valid]), method="spearman")),
            }
    masks = {
        "arrival_before_reported_takeoff": use.flown_arrival_minus_takeoff_s.lt(0),
        "remaining_airborne_under_10m": use.flown_arrival_minus_takeoff_s.between(0, 600),
        "remaining_airborne_over_24h": use.flown_arrival_minus_takeoff_s.gt(86400),
        "flown_duration_negative": use.flown_duration_s.lt(0),
        "flown_duration_over_24h": use.flown_duration_s.gt(86400),
        "duration_plan_disagreement_over_1h": use.flown_minus_planned_duration_s.abs().gt(3600),
        "duration_plan_disagreement_over_3h": use.flown_minus_planned_duration_s.abs().gt(10800),
    }
    out["observable_regimes"] = {}
    for name, mask in masks.items():
        values = r[mask.to_numpy()]
        out["observable_regimes"][name] = {"rows": len(values), "fraction_of_centered_proxy_sse": float(np.sum(values*values)/np.sum(r*r)), "rmse_seconds": float(np.sqrt(np.mean(values*values))) if len(values) else None}
    return out


unused_clocks_columns = [
    "flown_arrival_minus_takeoff_s", "flown_arrival_minus_plan_s", "flown_duration_s",
    "flown_lobt_duration_s", "flown_minus_planned_duration_s", "flown_minus_initial_duration_s",
]


def main():
    destination = ROOT / "docs/next_lever/flown_clock_audit.json"
    if destination.exists():
        raise ValueError("Preserve the original audit.")
    base = ROOT / "data/nmclock/72e86725e23367ede5b5"
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    folds = {fold: pd.read_parquet(parent / "splits" / f"{fold}.parquet") for fold in ["seasonal", "forward"]}
    fitting = set.intersection(*[set(f.loc[f.partition.eq("fit"), ID]) for f in folds.values()])
    excluded = set.union(*[set(f.loc[~f.partition.eq("fit"), ID]) for f in folds.values()])
    assert not fitting.intersection(excluded)
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(assignment[ID].isin(fitting), "fit", "excluded")
    features, truth = load_partition(base, parent, catalog, assignment, "fit", BASE)
    sources = sorted(truth.source_file.unique())
    data = []
    inputs = {}
    for source in sources:
        path = ROOT / "data/raw" / source
        raw = pd.read_parquet(path, columns=RAW)
        raw = raw.loc[raw[ID].isin(fitting)].reset_index(drop=True)
        assert raw[PHASE].eq("DEP").all() and not raw[TIME].dt.month.eq(12).any()
        data.append(unused_clocks(raw))
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    clocks = pd.concat(data, ignore_index=True)
    joined = features.merge(clocks, on=ID, validate="one_to_one").merge(truth[[ID, TIME, TARGET]], on=ID, validate="one_to_one")
    assert len(joined) == len(fitting) == 1226945
    offset = np.where(joined.nmclock_available.eq(1), joined.nmclock_proxy_s, joined.conditional_schedule_offset_s)
    joined["residual"] = joined[TARGET].to_numpy()-offset
    groups = {"all": summary(joined)}
    for airport, group in joined.groupby("airport", sort=True, observed=True):
        groups[f"airport/{airport}"] = summary(group)
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=RAW)
    ranking = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    ranking_features = unused_clocks(ranking)
    ranking_features["nmclock_available"] = ranking["AOBT_3_flt"].notna().astype(int)
    report = {
        "created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
        "feature_bundle_id": manifest["bundle_id"], "raw_inputs_sha256": inputs,
        "fitting_rows": len(joined), "fitting_ids_sha256": identity_hash(joined[ID]),
        "actual_fitting_months": sorted(joined[TIME].dt.strftime("%Y-%m").unique()),
        "groups": groups, "ranking_covariate_coverage": summary(ranking_features),
        "split_file_sha256": {fold: sha256_file(parent / "splits" / f"{fold}.parquet") for fold in folds},
        "raw_columns": RAW, "audit_used": False, "models_fitted": 0,
        "interpretation": "Descriptive fitting-only arithmetic residuals, not v6 model errors or validation scores. Ranking is covariates only. Correlation does not establish incremental predictive value; any model test requires a new declaration.",
    }
    write_json(destination, report)
    print({"fitting_rows": len(joined), "overall": groups["all"], "ranking_covariate_coverage": report["ranking_covariate_coverage"]})


if __name__ == "__main__":
    main()
