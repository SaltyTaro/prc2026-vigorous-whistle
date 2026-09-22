"""Independently verify an adsb_v26 export: declaration, selection, v9 provenance, the saved residual models, per-month assignment, blend, floor, fixed rows.

Checks: declaration parents and the export receipt's provenance; the selection belongs to the declaration and selected the exported (eligible, non-check)
candidate; v9 reproduced from scratch (frozen v7, January-November mixture, equal to the uploaded v9 after rounding) and, on clock-present rows, equal to
max(twelve-month v7, floor) with the mirrored nodes re-predicting a sample of those rows exactly; the export's v9 equal to v22's recorded ranking v9; the
residual recomputed from every saved model (LightGBM boosters / CatBoost models reloaded) on an independently rebuilt ranking feature table (frozen feature
functions, the declared match directory), each ranking row taking its own calendar month's model for weighted members and the members averaged; the blend
recomputed with plain arithmetic from the twelve-month ranking tables and equal to the export after rounding; every row outside the residual scope equal to
the uploaded v23; the LIRF fill population and missing-clock LIRF rows equal to the uploaded values; the inherited members equal to the shipped v23 models by hash;
the residual training rows and the sample-weight sums recounted from the scratch tables, scopes and the declared distance schemes. Nothing is uploaded.
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
from run_adsb_v24 import ALL_MONTHS
from run_adsb_v25 import catboost_matrix, month_distance
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population

RANKING_MONTHS = ["2026-01", "2026-07"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v26_v1.json"))
    parser.add_argument("--selection", type=Path, default=Path("runs/adsb_v26/dev/selection.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v26/verification.json"))
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
    members = candidate["residual"] if isinstance(candidate["residual"], list) else [candidate["residual"]]
    if not selection["candidates"][name]["eligible"] or name in config["check_candidates"] or prov["residual"]["members"] != members:
        raise ValueError("The exported candidate was not eligible or its members differ from the receipt.")
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
    row_month = departures[TIME].dt.strftime("%Y-%m").to_numpy()
    # v9 from scratch (frozen v7, January-November mixture)
    mixture_config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    model, component, summary = fit_population_model(movements([f"2025-{m:02}" for m in range(1, 12)]), mixture_config)
    floor = summary["floor_seconds"]
    v7_frozen = pd.read_parquet(ROOT / config["parents"]["v7_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    _, _, _, interval, mixture, index = score_population(model, component, ranking_raw, ids, mixture_config)
    v9_frozen, _ = apply_candidate(v7_frozen, index, interval, mixture, mixture_config["candidates"]["mixture_75"]["mixture_weight"], mixture_config["mixture"]["schedule_interval_threshold_seconds"], floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    uploaded_v23 = pd.read_parquet(ROOT / config["parents"]["v23_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    v23_fit = read_json(ROOT / config["parents"]["v23_fit_record"]["path"])
    if not np.array_equal(np.rint(v9_frozen).astype(np.int32), uploaded_v9):
        raise ValueError("v9 does not reproduce from scratch.")
    pop_index = np.zeros(len(ids), dtype=bool)
    pop_index[index] = True
    if not np.array_equal(pop_index, population):
        raise ValueError("The mixture's population differs from the row-context population.")
    # Twelve-month v7: record, mirrored nodes re-predict a sample
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
    v22_values = pd.read_parquet(ROOT / config["parents"]["v22_ranking_values"]["path"]).set_index(ID).reindex(ids)
    export_dir = ROOT / "runs/adsb_v26" / f"export_v{args.version}"
    saved_values = pd.read_parquet(export_dir / "ranking_values.parquet")
    if not np.array_equal(saved_values[ID].to_numpy(), ids) or float(np.max(np.abs(saved_values["v9"].to_numpy(dtype=float) - v9))) > 1e-9 or float(np.max(np.abs(v22_values["v9"].to_numpy(dtype=float) - v9))) > 1e-9:
        raise ValueError("The export's v9 differs from the independently derived v9 or from v22's.")
    # Residual from the saved models on an independently rebuilt table
    matches_root = ROOT / config["matches_dirs"]["v11"]
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
    meta = read_json(export_dir / "residual_model.json")
    if prov["residual"]["record_sha256"] != sha256_file(export_dir / "residual_model.json") or meta["members"] != members or meta["scope"] != config["residual_scope"] or meta["matches_dir"] != config["matches_dirs"]["v11"]:
        raise ValueError("Residual record differs from the receipt or the selected candidate.")
    inside = scope_mask(table, config["residual_scope"])
    residual = np.zeros(len(ids))
    model_report = {}
    for member in members:
        inherited = member in config["inherited_variants"]["names"]
        variant = config["new_variants"].get(member, {})
        if inherited:
            targets = sorted(v23_fit["models"][member])
        else:
            targets = RANKING_MONTHS if variant.get("weights") else ["all"]
        if sorted(meta["models"][member]) != sorted(targets):
            raise ValueError(f"Member {member} has models for {sorted(meta['models'][member])}, expected {targets}.")
        part = np.zeros(len(ids))
        for target in targets:
            spec = meta["models"][member][target]
            path = ROOT / spec["file"]
            if sha256_file(path) != spec["sha256"] or prov["residual"]["models"][member][target]["sha256"] != spec["sha256"]:
                raise ValueError(f"Saved model {spec['file']} differs from its record or the receipt.")
            if inherited and (spec["sha256"] != v23_fit["models"][member][target]["sha256"] or spec["file"] != config["inherited_variants"]["shipped_models"][member][target]):
                raise ValueError(f"Inherited model {member}/{target} is not the shipped v23 model.")
            rows = inside & (np.ones(len(ids), dtype=bool) if target == "all" else row_month == target)
            x, _ = prepare_matrix(table[rows], config, spec["categories"], spec["feature_set"])
            if spec["kind"] == "lightgbm":
                booster = lgb.Booster(model_file=str(path))
                if booster.num_trees() != spec["rounds"]:
                    raise ValueError(f"Saved booster {spec['file']} has {booster.num_trees()} trees, recorded {spec['rounds']}.")
                part[rows] = booster.predict(x, num_iteration=spec["rounds"])
            else:
                from catboost import CatBoostRegressor
                cb = CatBoostRegressor()
                cb.load_model(str(path))
                if cb.tree_count_ != spec["rounds"]:
                    raise ValueError(f"Saved CatBoost model {spec['file']} has {cb.tree_count_} trees, recorded {spec['rounds']}.")
                part[rows] = cb.predict(catboost_matrix(x, config["residual_model"]["categorical"]))
            model_report[f"{member}/{target}"] = {"kind": spec["kind"], "rounds": spec["rounds"], "rows_predicted": int(rows.sum())}
        member_delta = float(np.max(np.abs(part - saved_values[f"residual:{member}"].to_numpy(dtype=float) * inside)))
        if member_delta > 1e-6:
            raise ValueError(f"Member {member} residual recomputation differs by {member_delta}.")
        residual += part / len(members)
    residual_delta = float(np.max(np.abs(residual - saved_values["residual"].to_numpy(dtype=float))))
    if residual_delta > 1e-6:
        raise ValueError(f"Residual recomputation differs by {residual_delta}.")
    # Blend with plain arithmetic from the twelve-month ranking tables
    values = np.where(inside, v9 + residual, v9)
    blends = [config["blend_with"], *config["extra_blends"]]
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
    if not np.array_equal(values_exported[~inside], uploaded_v23[~inside]):
        raise ValueError("Rows outside the residual scope differ from the uploaded v23.")
    if not np.array_equal(values_exported[population], uploaded_v9[population]) or not np.array_equal(values_exported[lirf_missing_clock], uploaded_v23[lirf_missing_clock]):
        raise ValueError("Fixed rows (LIRF fill population, missing-clock LIRF rows) differ from the uploaded values.")
    if values_exported.min() < floor:
        raise ValueError("Export below the floor.")
    # Residual training rows and weight sums recounted from the scratch tables, the scope and the declared distance schemes
    month_rows = {}
    for m in ALL_MONTHS:
        frame = pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / f"{m}.parquet", columns=[ID, "airport"])
        matched_ids = set(pd.read_parquet(matches_root / f"matches2_{m}.parquet", columns=[ID])[ID].tolist())
        month_rows[m] = int((frame["airport"].ne("LIRF") & frame[ID].isin(matched_ids)).sum())
    counted = sum(month_rows.values())
    weight_report = {}
    for member in members:
        variant = config["new_variants"].get(member) or read_json(ROOT / config["parents"]["v25_declaration"]["path"])["residual_variants"][member]
        for target, spec in meta["models"][member].items():
            if spec["training_rows"] != counted:
                raise ValueError(f"Residual training rows recount {counted} vs recorded {spec['training_rows']} for {member}/{target}.")
            if variant.get("weights"):
                scheme = variant["weights"]
                d = month_distance(np.array(ALL_MONTHS), target)
                expected = float(sum(month_rows[m] * float(scheme.get(str(int(k)), scheme["default"])) for m, k in zip(ALL_MONTHS, d)))
                if abs(expected - spec["weight_sum"]) > 1e-6:
                    raise ValueError(f"Weight sum recount {expected} vs recorded {spec['weight_sum']} for {member}/{target}.")
                weight_report[f"{member}/{target}"] = {"weight_sum": expected, "rows": counted}
            elif abs(spec["weight_sum"] - counted) > 1e-6:
                raise ValueError(f"Unweighted member {member} records weight sum {spec['weight_sum']} for {counted} rows.")
    report = {"verified_at_utc": utc_now(), "file": export.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "rows": int(len(ids)), "ids_sha256": identity_hash(pd.Series(ids)), "config_sha256": object_hash(config),
              "selected_candidate": name, "members": members, "models": model_report, "twelve_month_v7": {"run_sha256": sha256_file(run / "run.json"), "sampled_rows": int(len(sample)), "max_abs_difference": node_delta},
              "v9_reproduced_from_scratch": True, "v9_equal_to_v22": True, "residual_max_abs_difference": residual_delta, "residual_training_rows_recounted": counted, "training_rows_by_month": month_rows,
              "weight_sums": weight_report, "independent_recomputation_matches": True, "rows_outside_scope_equal_v23": True, "fixed_rows_equal_uploaded": True,
              "rows_in_scope": int(inside.sum()), "rows_touched": int(touched.sum()), "rows_changed_vs_v23": int((values_exported != uploaded_v23).sum()), "floor_seconds": floor,
              "matched_rows": int(table["adsb_matched"].eq(1).sum()), "match_tables": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in RANKING_MONTHS}, "stands_sha256": sha256_file(matches_root / "stands.parquet")}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["file", "rows", "selected_candidate", "residual_max_abs_difference", "independent_recomputation_matches", "rows_outside_scope_equal_v23", "fixed_rows_equal_uploaded", "rows_in_scope", "rows_changed_vs_v23", "residual_training_rows_recounted"]},
          "v7 sample max diff", node_delta)


if __name__ == "__main__":
    main()
