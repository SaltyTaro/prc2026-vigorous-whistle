"""Independently verify a LIRF-mixture export from raw data, the saved classifier and frozen parents.

Recomputes the population, the turnaround features (by per-row filtering rather
than sorted search), the normal component and floor from refit-month labels, the
classifier probabilities from the saved booster, and every changed row; checks
that untouched rows equal the uploaded v7 integers. Never uploads.
"""
from pathlib import Path
import argparse
import re
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_submission.contract import validate_submission


def load_months(months: list[str], columns: list[str]) -> pd.DataFrame:
    frames = []
    for month in months:
        m = int(month[-2:])
        if m == 12:
            raise ValueError("December remains reserved.")
        frames.append(pd.read_parquet(ROOT / "data/raw" / f"training_2025-{m:02}-01_2025-{m + 1:02}-01.parquet", columns=columns))
    return pd.concat(frames, ignore_index=True)


def independent_features(frame: pd.DataFrame, margin: float) -> pd.DataFrame:
    dep = frame[(frame[PHASE] == "DEP") & (frame["ADEP_mvt"] == "LIRF") & frame["AOBT_3_flt"].isna() & frame["LOBT_flt"].isna()
                & frame["AIRCRAFT_TYPE_mvt"].isna()].copy()
    arr = frame[(frame[PHASE] == "ARR") & (frame["ADES_mvt"] == "LIRF") & frame["STAND_mvt"].notna() & frame["BLOCK_TIME_UTC_mvt"].notna()]
    prev, nxt = [], []
    for stand, takeoff in zip(dep["STAND_mvt"], dep[TIME]):
        same = arr.loc[arr["STAND_mvt"] == stand, "BLOCK_TIME_UTC_mvt"] if isinstance(stand, str) else pd.Series([], dtype="datetime64[ns, UTC]")
        before = same[same < takeoff - pd.Timedelta(seconds=margin)]
        after = same[same >= takeoff - pd.Timedelta(seconds=margin)]
        prev.append(before.max() if len(before) else pd.NaT)
        nxt.append(after.min() if len(after) else pd.NaT)
    dep["prev"] = pd.to_datetime(pd.Series(prev, index=dep.index), utc=True)
    dep["next"] = pd.to_datetime(pd.Series(nxt, index=dep.index), utc=True)
    sec = lambda td: td.dt.total_seconds().to_numpy(dtype=float)
    local, sloc = dep[TIME].dt.tz_convert("Europe/Rome"), dep["SCHED_TIME_UTC_mvt"].dt.tz_convert("Europe/Rome")
    flight = dep["FLIGHT_mvt"].fillna("").astype(str)
    interval = sec(dep[TIME] - dep["SCHED_TIME_UTC_mvt"])
    return pd.DataFrame({
        ID: dep[ID].to_numpy(), "schedule_interval_s": interval, "signed_log_schedule_interval": np.sign(interval) * np.log1p(np.abs(interval)),
        "local_hour": (local.dt.hour + local.dt.minute / 60).to_numpy(), "scheduled_local_hour": (sloc.dt.hour + sloc.dt.minute / 60).to_numpy(),
        "date_shift_days": (local.dt.normalize() - sloc.dt.normalize()).dt.days.to_numpy(dtype=float), "day_of_week": local.dt.dayofweek.to_numpy(dtype=float),
        "month": local.dt.month.to_numpy(dtype=float), "flight_missing": flight.eq("").to_numpy(dtype=float),
        "flight_doubled_prefix": np.array([float(len(s) >= 5 and s[1:3] == s[3:5]) for s in flight]),
        "flight_alpha_tail": np.array([float(bool(re.search(r"[A-Z]$", s))) for s in flight]),
        "schedule_minus_previous_inblock_s": sec(dep["SCHED_TIME_UTC_mvt"] - dep["prev"]), "takeoff_minus_previous_inblock_s": sec(dep[TIME] - dep["prev"]),
        "next_inblock_minus_takeoff_s": sec(dep["next"] - dep[TIME]),
        "flight_prefix": flight.str[:3].where(flight.ne(""), "?").to_numpy(), "stand_prefix": dep["STAND_mvt"].fillna("?").astype(str).str[:1].to_numpy(),
        "runway": dep["RUNWAY_mvt"].fillna("?").astype(str).to_numpy(), "destination_region": dep["ADES_mvt"].fillna("??").astype(str).str[:2].to_numpy(),
        TARGET: dep[TARGET].to_numpy(dtype=float) if TARGET in dep else np.nan})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/lirf_mixture_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/lirf_mixture/verification.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = read_json(output.with_suffix(".receipt.json"))
    if receipt["sha256"] != sha256_file(output) or receipt["provenance"]["config_sha256"] != object_hash(config):
        raise ValueError("Export receipt does not match the file or declaration.")
    for name, parent in config["parents"].items():
        if sha256_file(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent changed: {name}")
    contract = validate_submission(output, ROOT / "data/raw/submitting.parquet")
    run_dir = ROOT / "runs/lirf_mixture" / f"export_v{args.version}"
    if sha256_file(run_dir / "classifier.txt") != receipt["provenance"]["classifier_sha256"]:
        raise ValueError("Saved classifier differs from the export receipt.")
    saved = read_json(run_dir / "classifier.json")
    spec = config["candidates"][receipt["provenance"]["selected_candidate"]]
    margin = config["turnaround_features"]["previous_arrival_margin_seconds"]
    columns = [ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt", "BLOCK_TIME_UTC_mvt", TARGET]
    # Refit-month parameters from fitting labels only.
    fitting = load_months(config["refit_months"], columns)
    fit_features = independent_features(fitting, margin)
    labels, interval = fit_features[TARGET].to_numpy(), fit_features["schedule_interval_s"].to_numpy()
    tol, (low, high) = config["fill_tolerance_seconds"], config["mixture"]["one_day_label_interval_seconds"]
    modes = np.select([np.abs(labels - interval) <= tol, (labels >= low) & (labels <= high)], [0, 2], 1)
    bands = [float(v) for v in config["mixture"]["normal_component_bands_seconds"]]
    band_of = lambda s: np.digitize(s, bands[1:-1], right=True)
    medians = {}
    for b in range(len(bands) - 1):
        chosen = labels[(modes == 1) & (band_of(interval) == b)]
        medians[str(b)] = float(np.median(chosen)) if len(chosen) else None
    floor = float(fitting.loc[fitting[PHASE] == "DEP", TARGET].min())
    if medians != saved["normal_component"]["band_medians"] or floor != saved["floor_seconds"]:
        raise ValueError("Normal component or floor does not reproduce from fitting labels.")
    if {str(k): int((modes == k).sum()) for k in range(3)} != saved["mode_counts"] or len(fit_features) != receipt["provenance"]["fitting_summary"]["fitting_population_rows"]:
        raise ValueError("Fitting population or mode counts differ.")
    # Ranking rows, saved classifier, mixture and final values.
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=columns)
    rank_features = independent_features(ranking, margin)
    booster = lgb.Booster(model_file=str(run_dir / "classifier.txt"))
    matrix = rank_features[booster.feature_name()].copy()
    for name in config["classifier"]["categorical"]:
        matrix[name] = pd.Categorical(matrix[name].astype(str), categories=saved["categories"][name])
    probabilities = booster.predict(matrix)
    s = rank_features["schedule_interval_s"].to_numpy()
    overall = saved["normal_component"]["overall_median"]
    normal = np.array([medians[str(b)] if medians[str(b)] is not None else overall for b in band_of(s)])
    p_fill = probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, 1])
    mixture = p_fill * s + (1 - p_fill) * normal
    v7_raw = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"]
    v8 = pd.read_parquet(output).set_index(ID)[TARGET]
    v7 = pd.read_parquet(ROOT / "runs/submissions/vigorous-whistle_v7.parquet").set_index(ID)[TARGET]
    ids = v8.index.to_numpy()
    expected = v7_raw.reindex(ids).to_numpy(dtype=float).copy()
    position = pd.Series(np.arange(len(ids)), index=ids)
    idx = position.reindex(rank_features[ID].to_numpy()).to_numpy().astype(int)
    eligible = s <= config["mixture"]["schedule_interval_threshold_seconds"]
    expected[idx[eligible]] = (1 - spec["mixture_weight"]) * expected[idx[eligible]] + spec["mixture_weight"] * mixture[eligible]
    floored = expected < floor
    expected[floored] = floor
    if not np.array_equal(np.rint(expected).astype(np.int32), v8.to_numpy()):
        raise ValueError("Independent recomputation differs from the export.")
    touched = np.zeros(len(ids), dtype=bool)
    touched[idx[eligible]] = True
    touched |= floored
    if not np.array_equal(v8.to_numpy()[~touched], v7.reindex(ids).to_numpy()[~touched]):
        raise ValueError("Rows outside the declared rules differ from v7.")
    saved_pop = pd.read_parquet(run_dir / "population_predictions.parquet").set_index(ID)
    max_prob_diff = float(np.max(np.abs(saved_pop.loc[rank_features[ID], "p_fill"].to_numpy() - probabilities[:, 0])))
    report = {"verified_at_utc": utc_now(), "file": output.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "contract": contract,
              "independent_recomputation_matches": True, "untouched_rows_equal_v7": True, "parameters_reproduce_from_fitting_labels": True,
              "population_rows": int(len(rank_features)), "mixture_rows": int(eligible.sum()), "floored_rows": int(floored.sum()),
              "changed_rows": int((np.rint(expected).astype(np.int32) != v7.reindex(ids).to_numpy()).sum()),
              "max_saved_probability_difference": max_prob_diff, "classifier_rounds": saved["rounds"], "floor_seconds": floor}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["population_rows", "mixture_rows", "floored_rows", "changed_rows", "max_saved_probability_difference", "classifier_rounds"]})


if __name__ == "__main__":
    main()
