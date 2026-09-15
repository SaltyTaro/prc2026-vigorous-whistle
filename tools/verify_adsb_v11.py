"""Verify an ADS-B residual export: receipts, v9 reproduction, residual recomputation, out-of-scope rows, and an independent spot check.

Recomputes the float v9 from scratch, rebuilds the ranking feature table from
the saved matches and stands (with the neighbour context when the declaration
uses it), reloads the saved residual booster, recomputes every exported integer,
checks that rows outside the candidate's scope equal the uploaded v9, and
independently re-derives the surface segment of a random sample of matched
ranking departures from the raw surface parquet by per-row filtering. Never uploads.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_submission.contract import validate_submission
from export_adsb_v11 import feature_columns, ranking_table
from run_adsb_v11 import check_parents, matches_dir, prepare_matrix, scope_mask
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population

SURFACE = ROOT / "data/external/adsblol/surface"


def independent_segment(surface: pd.DataFrame, icao: str, airport: str, takeoff: float) -> dict | None:
    """Plain re-derivation: ground points of this aircraft at this airport in the four hours before takeoff, after the last airborne point and the last gap."""
    t = surface[(surface["icao"] == icao) & (surface["airport"] == airport) & (surface["time"] >= takeoff - 4 * 3600) & (surface["time"] <= takeoff + 30)].sort_values("time")
    airborne = t[(~t["ground"]) & (t["time"] < takeoff - 60)]
    if len(airborne):
        t = t[t["time"] > airborne["time"].max()]
    g = t[t["ground"]]
    if len(g) < 3:
        return None
    times = g["time"].to_numpy()
    start = 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > 600:
            start = i
    seg = g.iloc[start:]
    if len(seg) < 3:
        return None
    return {"segment_start": float(seg["time"].iloc[0]), "last_ground": float(seg["time"].iloc[-1]), "points": int(len(seg))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v11_v1.json"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/adsb_v11"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v11/verification.json"))
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = read_json(output.with_suffix(".receipt.json"))
    if receipt["sha256"] != sha256_file(output) or receipt["provenance"]["config_sha256"] != object_hash(config):
        raise ValueError("Export receipt does not match the file or declaration.")
    check_parents(config)
    contract = validate_submission(output, ROOT / "data/raw/submitting.parquet")
    run_dir = ROOT / args.run_root / f"export_v{args.version}"
    if sha256_file(run_dir / "residual_model.txt") != receipt["provenance"]["residual_model_sha256"]:
        raise ValueError("Saved residual model differs from the export receipt.")
    saved = read_json(run_dir / "residual_model.json")
    weight = receipt["provenance"]["weight"]
    scope = receipt["provenance"].get("scope", "matched")
    feature_set = receipt["provenance"].get("feature_set")
    candidate = config["candidates"][receipt["provenance"]["selected_candidate"]]
    if scope != candidate.get("scope", "matched") or feature_set != candidate.get("features"):
        raise ValueError("Exported scope or feature set differs from the selected candidate's declaration.")
    mixture_config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    fitting = movements(config["refit_months"])
    v9_model, v9_component, v9_summary = fit_population_model(fitting, mixture_config)
    floor = v9_summary["floor_seconds"]
    if floor != saved["floor_seconds"]:
        raise ValueError("Floor differs from the saved model record.")
    ranking = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking.loc[ranking[PHASE].eq("DEP")].reset_index(drop=True)
    ids = departures[ID].to_numpy()
    v7 = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    _, _, _, interval, v9_mixture, index = score_population(v9_model, v9_component, ranking, ids, mixture_config)
    v9, _ = apply_candidate(v7, index, interval, v9_mixture, mixture_config["candidates"]["mixture_75"]["mixture_weight"], mixture_config["mixture"]["schedule_interval_threshold_seconds"], floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v9).astype(np.int32), uploaded_v9):
        raise ValueError("The uploaded v9 does not reproduce from scratch.")
    stands = pd.read_parquet(matches_dir(config) / "stands.parquet")
    table, matches = ranking_table(config, departures, stands, v9)
    columns = feature_columns(config)
    saved_features = pd.read_parquet(run_dir / "ranking_features.parquet").set_index(ID).reindex(ids)
    diff = np.nanmax(np.abs(table.set_index(ID)[columns].to_numpy(dtype=float) - saved_features[columns].to_numpy(dtype=float)))
    if not (diff == 0 or np.isnan(diff)):
        raise ValueError("Recomputed ADS-B features differ from the saved export features.")
    booster = lgb.Booster(model_file=str(run_dir / "residual_model.txt"))
    inside = scope_mask(table, scope)
    matched = table["adsb_matched"].eq(1).to_numpy()
    residual = np.zeros(len(table))
    x, _ = prepare_matrix(table[inside], config, saved["categories"], feature_set)
    if list(x.columns) != saved["features"] or booster.feature_name() != saved["features"]:
        raise ValueError("Feature list differs between the declaration, the saved model record and the booster.")
    residual[inside] = booster.predict(x)
    expected = np.maximum(v9 + weight * residual, floor)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(expected).astype(np.int32), exported):
        raise ValueError("Independent recomputation differs from the export.")
    if not np.array_equal(exported[~inside], uploaded_v9[~inside]):
        raise ValueError("Rows outside the scope differ from the uploaded v9.")
    max_res_diff = float(np.max(np.abs(residual - saved_features["residual"].to_numpy(dtype=float))))
    # Independent spot check of matched segments against the raw surface parquet.
    rng = np.random.default_rng(20260915)
    picks = rng.choice(np.flatnonzero(matched), size=min(args.sample, int(matched.sum())), replace=False)
    match_index = matches.set_index(ID)
    checked, agree = 0, 0
    cache = {}
    for i in picks:
        row = match_index.loc[ids[i]]
        day = pd.to_datetime(departures.loc[i, TIME]).strftime("%Y%m%d")
        if day not in cache:
            cache[day] = pd.read_parquet(SURFACE / f"surface_{day}.parquet")
        seg = independent_segment(cache[day], row["icao"], row["airport"], float(row["takeoff"]))
        checked += 1
        if seg is not None and abs(seg["segment_start"] - row["segment_start"]) < 1e-6 and abs(seg["last_ground"] - row["last_ground"]) < 1e-6 and seg["points"] == row["points"]:
            agree += 1
    if agree < 0.98 * checked:
        raise ValueError(f"Independent segment re-derivation agrees on only {agree}/{checked} sampled departures.")
    report = {"verified_at_utc": utc_now(), "file": output.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "contract": contract,
              "v9_reproduced_from_scratch": True, "features_reproduced": True, "independent_recomputation_matches": True, "rows_outside_scope_equal_v9": True,
              "scope": scope, "rows_in_scope": int(inside.sum()), "matched_rows": int(matched.sum()), "changed_rows_vs_v9": int((exported != uploaded_v9).sum()),
              "max_saved_residual_difference": max_res_diff, "segment_spot_check": {"sampled": checked, "agreeing": agree}, "weight": weight, "rounds": saved["rounds"], "floor_seconds": floor}
    if "v11_export" in config["parents"]:
        uploaded_v11 = pd.read_parquet(ROOT / config["parents"]["v11_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
        report["changed_rows_vs_v11"] = int((exported != uploaded_v11).sum())
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["scope", "rows_in_scope", "matched_rows", "changed_rows_vs_v9", "max_saved_residual_difference", "segment_spot_check"] + (["changed_rows_vs_v11"] if "changed_rows_vs_v11" in report else [])})


if __name__ == "__main__":
    main()
