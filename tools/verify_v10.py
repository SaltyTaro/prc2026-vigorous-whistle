"""Independently verify a v10 export from raw data, the saved classifiers and frozen parents.

Recomputes the population, turnaround features (per-row filtering), the density
features (plain per-day groupbys and a per-row window filter), the normal
component and floor from refit-month labels, the bagged probabilities from the
saved boosters, the clock blend from the mirrored refit predictions and the
negative-proxy routing from raw clocks, then checks every row of the export.
Never uploads.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, verify_receipts, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_submission.contract import validate_submission
from verify_lirf_mixture import independent_features, load_months

COLUMNS = [ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt", "BLOCK_TIME_UTC_mvt", TARGET]


def independent_density(frame: pd.DataFrame, population_ids: np.ndarray, spec: dict) -> pd.DataFrame:
    dep = frame[(frame[PHASE] == "DEP") & (frame["ADEP_mvt"] == "LIRF")].copy()
    dep["date"] = dep[TIME].dt.tz_convert(spec["timezone"]).dt.strftime("%Y-%m-%d")
    dep["s"] = (dep[TIME] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep["noclock"] = dep["AOBT_3_flt"].isna() & dep["LOBT_flt"].isna()
    dep["delayed"] = dep["s"] > spec["delayed_threshold_seconds"]
    pop = dep[dep[ID].isin(population_ids)].copy()
    out = []
    for _, row in pop.iterrows():
        same_day = dep[dep["date"] == row["date"]]
        pop_day = pop[pop["date"] == row["date"]]
        within = pop[(pop[TIME] >= row[TIME] - pd.Timedelta(seconds=spec["window_seconds"])) & (pop[TIME] <= row[TIME] + pd.Timedelta(seconds=spec["window_seconds"]))]
        out.append({ID: row[ID], "day_departures": float(len(same_day)), "day_noclock_departures": float(same_day["noclock"].sum()),
                    "day_delayed_share": float(same_day["delayed"].mean()), "day_population_rows": float(len(pop_day)),
                    "day_population_mean_s": float(pop_day["s"].mean()), "window2h_population_rows": float(len(within))})
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/v10_v2.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/v10/verification.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    mixture_config = read_json(ROOT / config["lirf_mixture_config"])
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = read_json(output.with_suffix(".receipt.json"))
    if receipt["sha256"] != sha256_file(output) or receipt["provenance"]["config_sha256"] != object_hash(config):
        raise ValueError("Export receipt does not match the file or declaration.")
    for name, parent in config["parents"].items():
        if sha256_file(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent changed: {name}")
    items = receipt["provenance"]["items"]
    contract = validate_submission(output, ROOT / "data/raw/submitting.parquet")
    run_dir = ROOT / "runs/v10" / f"export_v{args.version}"
    saved = read_json(run_dir / "classifier.json")
    if sha256_file(run_dir / "classifier.json") != receipt["provenance"]["classifier_record_sha256"]:
        raise ValueError("Saved classifier record differs from the export receipt.")
    for member in saved["members"]:
        if sha256_file(run_dir / member["file"]) != member["sha256"] or receipt["provenance"]["classifier_members_sha256"][str(member["seed"])] != member["sha256"]:
            raise ValueError("A saved classifier member differs from the export receipt.")
    if saved["seeds"] != config["classifier_bagging"]["seeds"] or len(saved["members"]) != config["classifier_bagging"]["members"]:
        raise ValueError("Saved members do not follow the declared bagging.")
    margin = mixture_config["turnaround_features"]["previous_arrival_margin_seconds"]
    # Refit-month parameters from fitting labels only.
    fitting = load_months(config["refit_months"], COLUMNS)
    fit_features = independent_features(fitting, margin)
    labels, interval = fit_features[TARGET].to_numpy(), fit_features["schedule_interval_s"].to_numpy()
    tol, (low, high) = mixture_config["fill_tolerance_seconds"], mixture_config["mixture"]["one_day_label_interval_seconds"]
    modes = np.select([np.abs(labels - interval) <= tol, (labels >= low) & (labels <= high)], [0, 2], 1)
    bands = [float(v) for v in mixture_config["mixture"]["normal_component_bands_seconds"]]
    band_of = lambda s: np.digitize(s, bands[1:-1], right=True)
    medians = {}
    for b in range(len(bands) - 1):
        chosen = labels[(modes == 1) & (band_of(interval) == b)]
        medians[str(b)] = float(np.median(chosen)) if len(chosen) else None
    floor = float(fitting.loc[fitting[PHASE] == "DEP", TARGET].min())
    if medians != saved["normal_component"]["band_medians"] or floor != saved["floor_seconds"]:
        raise ValueError("Normal component or floor does not reproduce from fitting labels.")
    if {str(k): int((modes == k).sum()) for k in range(3)} != saved["mode_counts"] or len(fit_features) != saved["fitting_rows"]:
        raise ValueError("Fitting population or mode counts differ.")
    # Ranking rows: independent features, saved boosters, mixture.
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=COLUMNS)
    rank_features = independent_features(ranking, margin)
    density = independent_density(ranking, rank_features[ID].to_numpy(), config["density_features"])
    rank_features = rank_features.merge(density, on=ID, how="left", validate="one_to_one")
    probabilities = []
    for member in saved["members"]:
        booster = lgb.Booster(model_file=str(run_dir / member["file"]))
        if booster.feature_name() != saved["feature_names"]:
            raise ValueError("A saved member has unexpected features.")
        matrix = rank_features[booster.feature_name()].copy()
        for name in mixture_config["classifier"]["categorical"]:
            matrix[name] = pd.Categorical(matrix[name].astype(str), categories=member["categories"][name])
        probabilities.append(booster.predict(matrix))
    probabilities = np.mean(probabilities, axis=0)
    s = rank_features["schedule_interval_s"].to_numpy()
    overall = saved["normal_component"]["overall_median"]
    normal = np.array([medians[str(b)] if medians[str(b)] is not None else overall for b in band_of(s)])
    p_fill = probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, 1])
    mixture = p_fill * s + (1 - p_fill) * normal
    # Compose in the declared order from the frozen parents.
    exported = pd.read_parquet(output).set_index(ID)[TARGET]
    ids = exported.index.to_numpy()
    v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    v7 = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    fallback = pd.read_parquet(ROOT / config["parents"]["fallback_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    dep = ranking[ranking[PHASE] == "DEP"].set_index(ID).reindex(ids)
    aobt = (dep[TIME] - dep["AOBT_3_flt"]).dt.total_seconds().to_numpy(dtype=float)
    lobt = (dep[TIME] - dep["LOBT_flt"]).dt.total_seconds().to_numpy(dtype=float)
    proxy = np.where(np.isfinite(aobt), aobt, lobt)
    clock = np.isfinite(proxy)
    negative = clock & (proxy < 0)
    expected = v7.copy()
    position = pd.Series(np.arange(len(ids)), index=ids)
    idx = position.reindex(rank_features[ID].to_numpy()).to_numpy().astype(int)
    eligible = s <= mixture_config["mixture"]["schedule_interval_threshold_seconds"]
    weight = config["items"]["density_classifier"]["mixture_weight"]
    if "density_classifier" not in items:
        raise ValueError("This verifier expects the density classifier in the exported items.")
    expected[idx[eligible]] = (1 - weight) * expected[idx[eligible]] + weight * mixture[eligible]
    refit_info = None
    if "cat_long_25" in items:
        refit = receipt["provenance"]["cat_long_refit"]
        refit_dir = ROOT / "runs/azure_arcw0_v10" / refit["run_id"]
        run = read_json(refit_dir / "run.json")
        if sha256_file(refit_dir / "run.json") != refit["run_sha256"] or run["iterations"] != config["cat_long_refit"]["iterations"] or run["audit_used"]:
            raise ValueError("The cat_long refit record differs from the receipt.")
        verify_receipts(refit_dir, run["artifacts"])
        cat_long = pd.read_parquet(refit_dir / "ranking_predictions.parquet").set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(cat_long), clock):
            raise ValueError("cat_long predictions must cover exactly the clock rows.")
        share = config["items"]["cat_long_25"]["weight"]
        expected[clock] = (1 - share) * expected[clock] + share * cat_long[clock]
        refit_info = {"run_id": run["run_id"], "iterations": run["iterations"], "clock_rows": int(clock.sum()),
                      "max_model_reload_difference_seconds": run["max_model_reload_difference_seconds"]}
    if "route_negative_proxy" in items:
        expected[negative] = fallback[negative]
    floored = expected < floor
    expected[floored] = floor
    if not np.array_equal(np.rint(expected).astype(np.int32), exported.to_numpy()):
        raise ValueError("Independent recomputation differs from the export.")
    touched = clock.copy()
    touched[idx[eligible]] = True
    touched |= floored
    if not np.array_equal(exported.to_numpy()[~touched], v9[~touched]):
        raise ValueError("Rows outside the declared items differ from v9.")
    saved_pop = pd.read_parquet(run_dir / "population_predictions.parquet").set_index(ID)
    max_prob_diff = float(np.max(np.abs(saved_pop.loc[rank_features[ID], "p_fill"].to_numpy() - p_fill)))
    max_density_diff = float(np.max(np.abs(saved_pop.loc[rank_features[ID], config["density_features"]["names"]].to_numpy(dtype=float) - density.set_index(ID).loc[rank_features[ID]].to_numpy(dtype=float))))
    report = {"verified_at_utc": utc_now(), "file": output.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "contract": contract, "items": items,
              "independent_recomputation_matches": True, "untouched_rows_equal_v9": True, "parameters_reproduce_from_fitting_labels": True,
              "population_rows": int(len(rank_features)), "mixture_rows": int(eligible.sum()), "clock_rows": int(clock.sum()), "routed_rows": int(negative.sum()),
              "floored_rows": int(floored.sum()), "changed_rows_vs_v9": int((np.rint(expected).astype(np.int32) != v9).sum()),
              "max_saved_probability_difference": max_prob_diff, "max_saved_density_feature_difference": max_density_diff,
              "member_rounds": [m["rounds"] for m in saved["members"]], "floor_seconds": floor, "cat_long_refit": refit_info}
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["population_rows", "mixture_rows", "clock_rows", "routed_rows", "floored_rows", "changed_rows_vs_v9", "max_saved_probability_difference", "max_saved_density_feature_difference"]})


if __name__ == "__main__":
    main()
