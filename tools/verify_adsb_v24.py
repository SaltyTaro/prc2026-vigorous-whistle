"""Independently verify an adsb_v24 export: declaration, selection, twelve-month v7 record and mirrored nodes, v9 provenance, residual, blend, floor.

Checks: declaration parents and the export receipt's provenance; the selection belongs to the declaration and selected the exported candidate;
the twelve-month v7 record (no held-out month, twelve fit months, the frozen component counts, the ranking identity) and, from the mirrored
nodes, a re-prediction of sampled clock-present ranking rows equal to the saved ranking predictions; v9 reproduced from scratch (frozen v7,
January-November mixture) and equal to the uploaded v9 after rounding; the export's v9 equal to max(twelve-month v7, floor) on clock-present
rows and to the reproduced v9 elsewhere; the residual recomputed from the saved booster on an independently rebuilt ranking feature table
(frozen feature functions, the declared match directory, season_cos from the movement clock); the blend recomputed with plain arithmetic from
the twelve-month ranking tables and equal to the export after rounding; the LIRF fill population and missing-clock LIRF rows equal to the
uploaded values; the residual training rows recounted from the scratch tables and scopes. Nothing is uploaded.
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
from prc2026.validation import identity_hash
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features, neighbour_features
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_nmclock.features import route_mask
from prc2026_postprocess.rules import row_context
from prc2026_surface_airports.inference import airport_prediction
from prc2026_surface_refit.inference import WEIGHTS
from prc2026_timeline.features import FEATURE_NAMES
from prc2026_timeline.inference import node_prediction
from run_adsb_v11 import check_parents, prepare_matrix, scope_mask
from run_adsb_v24 import ALL_MONTHS, FRAME_OF
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population

RANKING_MONTHS = ["2026-01", "2026-07"]
SCOPES_USED = ["matched_no_lirf", "all_no_lirf", "lirf_nonpop_matched", "lirf_nonpop_unmatched"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v24_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_v24/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v24/verification.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    check_parents(config)
    selection = read_json(ROOT / args.selection)
    export = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = read_json(export.with_suffix(".receipt.json"))
    prov = receipt["provenance"]
    if receipt["sha256"] != sha256_file(export) or prov["config_sha256"] != object_hash(config) or prov["policy"] != config["prediction_postprocess"]:
        raise ValueError("Export receipt does not belong to this declaration.")
    if selection["config_sha256"] != object_hash(config) or prov["selection_sha256"] != sha256_file(ROOT / args.selection) or selection["selected_candidate"] != prov["selected_candidate"]:
        raise ValueError("The selection does not belong to this declaration or the export.")
    name = prov["selected_candidate"]
    candidate = config["candidates"][name]
    variant = config["residual_variants"][candidate["residual"]]
    if not selection["candidates"][name]["eligible"]:
        raise ValueError("The exported candidate was not eligible.")
    exported = pd.read_parquet(export)
    ids = exported[ID].to_numpy()
    values_exported = exported[TARGET].to_numpy()
    ranking_raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking_raw.loc[ranking_raw[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("Export rows are not the ranking departures in template order.")
    context = row_context(departures).set_index(ID).reindex(ids)
    clock = context["clock_available"].to_numpy(dtype=bool)
    population = context["hedge_population"].to_numpy(dtype=bool)
    # v9 from scratch (frozen v7, January-November mixture)
    mixture_config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    model, component, summary = fit_population_model(movements([f"2025-{m:02}" for m in range(1, 12)]), mixture_config)
    floor = summary["floor_seconds"]
    v7_frozen = pd.read_parquet(ROOT / config["parents"]["v7_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    _, _, _, interval, mixture, index = score_population(model, component, ranking_raw, ids, mixture_config)
    v9_frozen, _ = apply_candidate(v7_frozen, index, interval, mixture, mixture_config["candidates"]["mixture_75"]["mixture_weight"], mixture_config["mixture"]["schedule_interval_threshold_seconds"], floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    uploaded_v21 = pd.read_parquet(ROOT / config["parents"]["v21_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v9_frozen).astype(np.int32), uploaded_v9):
        raise ValueError("v9 does not reproduce from scratch.")
    pop_index = np.zeros(len(ids), dtype=bool)
    pop_index[index] = True
    if not np.array_equal(pop_index, population):
        raise ValueError("The mixture's population differs from the row-context population.")
    # Twelve-month v7: record, mirrored nodes re-predict a sample, v9 provenance
    v7_report = None
    v9 = v9_frozen.copy()
    if candidate["v9"] == "cross_fit":
        run = ROOT / "runs/adsb_v24/v7/refit"
        record = read_json(run / "run.json")
        nodes = read_json(run / "nodes.json")
        frozen_counts = read_json(ROOT / "configs/surface_refit_v1.json")["component_iterations"]
        if (record["stage"] != "refit_twelve_months" or record["held_out_months"] or record["fit_months"] != ALL_MONTHS or record["component_iterations"] != frozen_counts
                or record["predicted_ids_sha256"] != identity_hash(pd.Series(ids)) or record["seed"] != 20260913
                or {n: c["iterations"] for n, c in nodes["globals"].items()} != frozen_counts["globals"] or {a: c["iterations"] for a, c in nodes["airports"].items()} != frozen_counts["airports"]
                or prov["twelve_month_v7"]["run_sha256"] != sha256_file(run / "run.json") or prov["twelve_month_v7"]["predictions_sha256"] != sha256_file(run / "ranking_predictions.parquet")):
            raise ValueError("The twelve-month v7 record does not match the declaration, the frozen counts or the receipt.")
        saved = pd.read_parquet(run / "ranking_predictions.parquet").set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(saved), clock):
            raise ValueError("Twelve-month v7 predictions do not cover exactly the clock-present rows.")
        bundle = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
        features = pd.read_parquet(bundle / "features/ranking.parquet", columns=[ID, *FEATURE_NAMES])
        if not np.array_equal(features[ID].to_numpy(), ids) or not np.array_equal(route_mask(features), clock):
            raise ValueError("Ranking feature frame order or clock route differs.")
        rng = np.random.default_rng(config["seed"])
        sample = np.sort(rng.choice(np.flatnonzero(clock), size=min(args.sample, int(clock.sum())), replace=False))
        rows = features.iloc[sample].reset_index(drop=True)
        repredicted = np.zeros(len(rows))
        for comp, weight in WEIGHTS.items():
            repredicted += weight * (airport_prediction(run, nodes["airports"], rows, args.threads) if comp == "surface_airports" else node_prediction(run, nodes["globals"][comp], rows, args.threads))
        node_delta = float(np.max(np.abs(repredicted - saved[sample])))
        if node_delta > 1e-6:
            raise ValueError(f"Mirrored twelve-month v7 nodes differ from the saved predictions by {node_delta}.")
        v9 = np.where(clock, np.maximum(saved, floor), v9_frozen)
        v7_report = {"run_sha256": sha256_file(run / "run.json"), "clock_fit_rows": record["clock_fit_rows"], "sampled_rows": int(len(sample)), "max_abs_difference": node_delta,
                     "component_iterations": record["component_iterations"], "code_sha256": record["code"]["sha256"]}
    export_dir = ROOT / "runs/adsb_v24" / f"export_v{args.version}"
    saved_values = pd.read_parquet(export_dir / "ranking_values.parquet")
    if not np.array_equal(saved_values[ID].to_numpy(), ids) or float(np.max(np.abs(saved_values["v9"].to_numpy(dtype=float) - v9))) > 1e-9:
        raise ValueError("The export's v9 differs from the independently derived v9.")
    # Residual from the saved booster on an independently rebuilt table
    matches_root = ROOT / config["matches_dirs"][variant["matches_dir"]]
    stands = pd.read_parquet(matches_root / "stands.parquet")
    matches = pd.concat([pd.read_parquet(matches_root / f"matches2_{m}.parquet") for m in RANKING_MONTHS], ignore_index=True)
    table = departure_features(departures, matches, stands).merge(departures[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    table = table.merge(neighbour_features(departures, matches), on=ID, validate="one_to_one").rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    if not np.array_equal(table[ID].to_numpy(), ids):
        raise ValueError("Rebuilt ranking table order differs.")
    table["airport"] = departures["ADEP_mvt"].to_numpy()
    table["v9"] = v9
    table["proxy_s"] = context["proxy_s"].to_numpy()
    table["schedule_interval_s"] = context["schedule_interval_s"].to_numpy()
    table["population"] = population
    table["hour"] = departures[TIME].dt.hour + departures[TIME].dt.minute / 60
    table["season_cos"] = np.cos(2 * np.pi * (departures[TIME].dt.dayofyear.to_numpy(dtype=float) - 1) / 365.25)
    meta = read_json(export_dir / "residual_model.json")
    if prov["residual_model_sha256"] != sha256_file(export_dir / "residual_model.txt") or meta["feature_set"] != variant["feature_set"] or meta["scope"] != config["residual_scope"] or meta["matches_dir"] != config["matches_dirs"][variant["matches_dir"]]:
        raise ValueError("Residual booster or its record differs from the receipt or the selected variant.")
    booster = lgb.Booster(model_file=str(export_dir / "residual_model.txt"))
    inside = scope_mask(table, config["residual_scope"])
    x, _ = prepare_matrix(table[inside], config, meta["categories"], meta["feature_set"])
    residual = np.zeros(len(ids))
    residual[inside] = booster.predict(x, num_iteration=meta["rounds"])
    residual_delta = float(np.max(np.abs(residual - saved_values["residual"].to_numpy(dtype=float))))
    if residual_delta > 1e-9:
        raise ValueError(f"Residual recomputation differs by {residual_delta}.")
    # Blend with plain arithmetic from the twelve-month ranking tables
    values = np.where(inside, v9 + residual, v9)
    blends = [config["blend_with"], *config["extra_blends"], *candidate.get("extra_blends", [])]
    if blends != [prov["blend_with"], *prov["extra_blends"]]:
        raise ValueError("Blend list differs from the receipt.")
    touched = inside.copy()
    for blend in blends:
        partner = pd.read_parquet(ROOT / config["extra_feature_tables"][blend["name"]]["ranking"]).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        where = scope_mask(table, blend["scope"]) & np.isfinite(partner)
        values = np.where(where, (1 - blend["weight"]) * values + blend["weight"] * partner, values)
        touched |= where
    values = np.maximum(values, floor)
    recomputed = np.rint(values).astype(np.int32)
    if not np.array_equal(recomputed, values_exported):
        raise ValueError(f"Independent recomputation differs from the export on {int((recomputed != values_exported).sum())} rows.")
    lirf_missing_clock = table["airport"].eq("LIRF").to_numpy() & ~clock
    if not np.array_equal(values_exported[population], uploaded_v9[population]) or not np.array_equal(values_exported[lirf_missing_clock], uploaded_v21[lirf_missing_clock]):
        raise ValueError("Fixed rows (LIRF fill population, missing-clock LIRF rows) differ from the uploaded values.")
    if candidate["v9"] == "fold" and not np.array_equal(values_exported[~touched], uploaded_v9[~touched]):
        raise ValueError("Rows outside every scope differ from the uploaded v9.")
    if values_exported.min() < floor:
        raise ValueError("Export below the floor.")
    # Residual training rows recounted from the scratch tables and the scope (clock-present, matched, non-LIRF; finite v9)
    counted, months_used = 0, []
    for m in ALL_MONTHS:
        src = candidate["v9"] if m in FRAME_OF else "cross_fit"
        frame = pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / f"{m}.parquet") if src == "cross_fit" else pd.read_parquet(ROOT / "scratch" / f"{FRAME_OF[m]}.parquet")
        frame = frame[frame[TIME].dt.strftime("%Y-%m").eq(m)]
        month_matches = pd.read_parquet(matches_root / f"matches2_{m}.parquet")
        matched_ids = set(month_matches[ID].tolist())
        counted += int((frame["airport"].ne("LIRF") & frame[ID].isin(matched_ids)).sum())
        months_used.append(f"{m}:{src}")
    if counted != meta["training_rows"] or months_used != meta["training_months"]:
        raise ValueError(f"Residual training rows recount {counted} vs recorded {meta['training_rows']}; months {months_used} vs {meta['training_months']}.")
    report = {"verified_at_utc": utc_now(), "file": export.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "rows": int(len(ids)), "ids_sha256": identity_hash(pd.Series(ids)), "config_sha256": object_hash(config),
              "selected_candidate": name, "v9_provenance": candidate["v9"], "twelve_month_v7": v7_report, "v9_reproduced_from_scratch": True, "v9_rows_changed_vs_frozen": int((np.abs(v9 - v9_frozen) > 1e-9).sum()),
              "residual_max_abs_difference": residual_delta, "residual_rounds": meta["rounds"], "residual_training_rows": meta["training_rows"], "residual_training_rows_recounted": counted, "residual_months": meta["training_months"],
              "independent_recomputation_matches": True, "fixed_rows_equal_uploaded": True, "rows_in_scope": int(inside.sum()), "rows_touched": int(touched.sum()), "floor_seconds": floor,
              "matched_rows": int(table["adsb_matched"].eq(1).sum()), "match_tables": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in RANKING_MONTHS}, "stands_sha256": sha256_file(matches_root / "stands.parquet")}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["file", "rows", "selected_candidate", "residual_max_abs_difference", "independent_recomputation_matches", "fixed_rows_equal_uploaded", "rows_in_scope", "residual_training_rows_recounted"]}, "v7", v7_report and v7_report["max_abs_difference"])


if __name__ == "__main__":
    main()
